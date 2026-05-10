import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import math
import re
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import AgentRLDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model
from trainer.rollout_engine import create_rollout_engine
from trainer.huawei_adapter import get_device, device_count, get_amp_context, get_grad_scaler

warnings.filterwarnings('ignore')


def parse_tool_calls(text):
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    calls = []
    for m in matches:
        try:
            calls.append(json.loads(m.strip()))
        except json.JSONDecodeError:
            pass
    return calls


def simulate_tool_execution(tool_call, gt_data):
    name = tool_call.get('name', '')
    args = tool_call.get('arguments', {})
    func_to_gt = {}
    for item in gt_data:
        if isinstance(item, dict):
            func = item.get('function', item.get('name', ''))
            result = item.get('result', item.get('output', str(item)))
            func_to_gt[func] = str(result)
    return func_to_gt.get(name, str(args))


def calculate_tool_reward(response, gt_data):
    reward = 0.0
    calls = parse_tool_calls(response)
    if not calls and gt_data:
        return -0.5
    for call in calls:
        name = call.get('name', '')
        if name:
            reward += 0.3
            args = call.get('arguments', {})
            expected = None
            for item in gt_data:
                if isinstance(item, dict) and item.get('function', item.get('name', '')) == name:
                    expected = item.get('result', item.get('output', ''))
                    break
            if expected:
                reward += 0.2
    if calls:
        reward += 0.2 * min(1.0, len(calls) / max(len(gt_data), 1))
    return min(reward, 2.0)


def build_agent_prompt(messages, tools):
    prompt = ''
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        prompt += f'<|im_start|>{role}\n{content}<|im_end|>\n'
    return prompt


def agent_train_epoch(epoch, loader, iters, rollout_engine, start_step=0, wandb=None):
    for step, batch in enumerate(loader, start=start_step + 1):
        all_messages = batch['messages']
        all_tools = batch['tools']
        all_gt = batch['gt']

        prompts = []
        for messages, tools in zip(all_messages, all_tools):
            conv = messages.copy()
            if tools:
                system_content = ''
                for m in conv:
                    if m.get('role') == 'system':
                        system_content = m.get('content', '')
                        break
                if system_content:
                    pass
            prompt = build_agent_prompt(conv, tools)
            prompts.append(prompt)

        prompt_inputs = tokenizer(prompts, return_tensors='pt', padding=True, padding_side='left', add_special_tokens=False, truncation=True, max_length=args.max_seq_len).to(args.device)

        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs['input_ids'],
            attention_mask=prompt_inputs['attention_mask'],
            num_generations=1, max_new_tokens=args.max_gen_len, temperature=0.9
        )
        outputs = rollout_result.output_ids
        completion_ids = rollout_result.completion_ids
        completions = rollout_result.completions
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()
        prompt_lens = rollout_result.prompt_lens.to(args.device)
        full_mask = (outputs != tokenizer.pad_token_id).long()
        logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=args.device).unsqueeze(0)

        rewards = torch.zeros(len(completions), device=args.device)
        for i, (resp, gt) in enumerate(zip(completions, all_gt)):
            rewards[i] = calculate_tool_reward(resp, gt)

        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            res = model_unwrapped(outputs, attention_mask=full_mask)
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
        per_token_loss = -(torch.min(ratio * advantages.unsqueeze(1), clipped_ratio * advantages.unsqueeze(1)))
        policy_loss = ((per_token_loss * completion_pad_mask).sum(dim=1) / completion_pad_mask.sum(dim=1).clamp(min=1)).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), Reward: {rewards.mean().item():.4f}, Policy Loss: {policy_loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.8f}')
            if wandb and is_main_process():
                wandb.log({'reward': rewards.mean().item(), 'policy_loss': policy_loss.item(), 'lr': scheduler.get_last_lr()[0]})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            torch.save({k: v.half().cpu() for k, v in raw_model.state_dict().items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()

        if step % args.save_interval == 0 or step == iters:
            rollout_engine.update_policy(model)

        del prompt_inputs, outputs, completion_ids, completions, per_token_logps, rewards

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ascend MiniMind Agent RL')
    parser.add_argument('--save_dir', type=str, default='../out')
    parser.add_argument('--save_weight', default='agent', type=str)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--max_seq_len', default=1024, type=int)
    parser.add_argument('--max_gen_len', type=int, default=512)
    parser.add_argument('--data_path', type=str, default='../dataset/agent.jsonl')
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--from_weight', default='full_sft', type=str)
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1])
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='Ascend-MiniMind-Agent')
    parser.add_argument('--use_compile', default=0, type=int, choices=[0, 1])
    parser.add_argument('--rollout_engine', type=str, default='torch', choices=['torch', 'sglang'])
    args = parser.parse_args()

    if args.device is None:
        dev = get_device()
        args.device = f'{dev}:0' if device_count() > 0 else 'cpu'

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        dev = get_device()
        args.device = f'{dev}:{local_rank}'
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    device_type = get_device()
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    autocast_ctx = nullcontext() if device_type == 'cpu' else get_amp_context(dtype=dtype, device_type=device_type)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        wandb.init(project=args.wandb_project, id=wandb_id, resume='must' if wandb_id else None)

    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    rollout_engine = create_rollout_engine(engine_type=args.rollout_engine, policy_model=model, tokenizer=tokenizer, device=args.device, autocast_ctx=autocast_ctx)
    train_ds = AgentRLDataset(args.data_path, tokenizer, max_length=args.max_seq_len + args.max_gen_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = math.ceil(len(train_ds) / args.batch_size / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.learning_rate / 10)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    if args.use_compile == 1:
        model = torch.compile(model)
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine.update_policy(model)

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=(device_type == 'cuda'))
        agent_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, start_step, wandb)

    if dist.is_initialized():
        dist.destroy_process_group()
