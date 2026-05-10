import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math
import time
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine
from trainer.huawei_adapter import get_device, device_count, get_amp_context, get_grad_scaler

warnings.filterwarnings('ignore')


def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        values = self.value_head(hidden_states).squeeze(-1)
        return values


def calculate_rewards(prompts, responses, reward_model):
    rewards = torch.zeros(len(responses), device=args.device)
    with torch.no_grad():
        reward_model_scores = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]
            answer = response
            rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
            if '</think>' in response:
                thinking_content, answer_content = response.split('</think>', 1)
                rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                rewards[i] += 0.25 if response.count('</think>') == 1 else -0.25
                answer = answer_content.strip()
            rewards[i] -= rep_penalty(answer)
            score = reward_model.get_score(messages, answer)
            reward_model_scores.append(score)
        rewards += torch.tensor(reward_model_scores, device=args.device)
    return rewards


def ppo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step=0, wandb=None):
    actor_model.train()
    critic_model.train()
    grad_accum_step = 0
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len, padding_side="left").to(args.device)
        rollout_result = rollout_engine.rollout(prompt_ids=enc.input_ids, attention_mask=enc.attention_mask, num_generations=1, max_new_tokens=args.max_gen_len, temperature=0.8)
        gen_out = rollout_result.output_ids
        completion_ids = rollout_result.completion_ids
        prompt_lens = rollout_result.prompt_lens.to(args.device)
        responses_text = rollout_result.completions
        old_resp_logp = rollout_result.per_token_logps.to(args.device)
        rewards = calculate_rewards(prompts, responses_text, reward_model)

        full_mask = (gen_out != tokenizer.pad_token_id).long()
        labels = gen_out[:, 1:].clone()
        resp_labels = completion_ids
        resp_idx = torch.arange(resp_labels.size(1), device=gen_out.device).unsqueeze(0)
        logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx
        resp_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        resp_lengths = resp_pad_mask.sum(dim=1)
        valid_resp = resp_lengths > 0
        eos_mask = resp_labels.eq(tokenizer.eos_token_id) & resp_pad_mask
        has_eos = eos_mask.any(dim=1)
        eos_pos = torch.argmax(eos_mask.int(), dim=1)
        resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths).long().clamp(min=1)
        resp_policy_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()
        resp_value_mask = resp_policy_mask.clone()

        with torch.no_grad():
            critic_unwrapped = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
            values_seq = critic_unwrapped(input_ids=gen_out, attention_mask=full_mask)
            old_resp_values = values_seq.gather(1, logp_pos) * resp_value_mask
            ref_resp_logp = F.log_softmax(ref_model(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
            token_rewards = torch.zeros_like(old_resp_logp)
            last_idx = resp_lengths - 1
            token_rewards[torch.arange(rewards.size(0), device=args.device)[valid_resp], last_idx[valid_resp]] += rewards[valid_resp]
            gen_len = old_resp_values.size(1)
            lastgaelam = torch.zeros(rewards.size(0), device=args.device)
            advs_rev = []
            for t in reversed(range(gen_len)):
                nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
                delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
                lastgaelam = delta + args.gamma * args.lam * lastgaelam
                advs_rev.append(lastgaelam)
            advantages = torch.stack(advs_rev[::-1], dim=1)
            returns = advantages + old_resp_values
            adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            adv_var = ((advantages - adv_mean) ** 2 * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask

        mb_size = max(1, min(args.mini_batch_size, rewards.size(0)))
        stop_ppo = False
        policy_loss_sum = value_loss_sum = kl_sum = kl_ref_sum = clipfrac_sum = aux_loss_sum = 0.0
        log_count = 0
        actor_unwrapped = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
        critic_unwrapped = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
        for ppo_epoch in range(args.ppo_update_iters):
            if stop_ppo: break
            b_inds = torch.randperm(rewards.size(0), device=args.device)
            for i in range(0, rewards.size(0), mb_size):
                inds = b_inds[i:i + mb_size]
                mb_values_seq = critic_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                mb_resp_values = mb_values_seq.gather(1, logp_pos[inds])
                with autocast_ctx:
                    res = actor_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                    aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
                mb_resp_logp = F.log_softmax(res.logits[:, :-1], dim=-1).gather(2, labels[inds].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos[inds])
                log_ratio = mb_resp_logp - old_resp_logp[inds]
                approx_kl = (0.5 * (log_ratio ** 2) * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                approx_kl_val = approx_kl.detach().clone()
                if dist.is_initialized():
                    dist.all_reduce(approx_kl_val, op=dist.ReduceOp.AVG)
                if approx_kl_val > args.early_stop_kl:
                    stop_ppo = True
                ratio = torch.exp(log_ratio)
                clipfrac = ((((ratio - 1.0).abs() > args.clip_epsilon).float() * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1))
                kl_ref_penalty = ((torch.exp(ref_resp_logp[inds] - mb_resp_logp) - (ref_resp_logp[inds] - mb_resp_logp) - 1.0) * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                policy_loss = ((torch.max(-advantages[inds] * ratio, -advantages[inds] * torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon)) * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1) + args.kl_coef * kl_ref_penalty)
                value_loss = 0.5 * (torch.max((mb_resp_values - returns[inds]) ** 2, (torch.clamp(mb_resp_values, old_resp_values[inds] - args.cliprange_value, old_resp_values[inds] + args.cliprange_value) - returns[inds]) ** 2) * resp_value_mask[inds]).sum() / resp_value_mask[inds].sum().clamp(min=1)
                kl = approx_kl_val
                kl_ref = kl_ref_penalty.detach()
                loss = (policy_loss + args.vf_coef * value_loss + aux_loss) * (0.0 if stop_ppo else 1.0) / args.accumulation_steps
                loss.backward()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                kl_sum += kl.item()
                kl_ref_sum += kl_ref.item()
                clipfrac_sum += clipfrac.item()
                aux_loss_sum += aux_loss.item()
                log_count += 1
                grad_accum_step += 1
                if grad_accum_step % args.accumulation_steps == 0:
                    clip_grad_norm_(actor_model.parameters(), args.grad_clip)
                    clip_grad_norm_(critic_model.parameters(), args.grad_clip)
                    actor_optimizer.step()
                    critic_optimizer.step()
                    actor_scheduler.step()
                    critic_scheduler.step()
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()
        if grad_accum_step % args.accumulation_steps != 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

        if step % args.save_interval == 0 or step == iters:
            rollout_engine.update_policy(actor_model)

        if is_main_process():
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), Reward: {rewards.mean().item():.4f}, KL_ref: {kl_ref_sum / max(log_count, 1):.4f}, Critic: {value_loss_sum / max(log_count, 1):.4f}')
            if wandb:
                wandb.log({"reward": rewards.mean().item(), "kl_ref": kl_ref_sum / max(log_count, 1)})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            torch.save({k: v.half().cpu() for k, v in raw_actor.state_dict().items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            actor_model.train()

        del enc, gen_out, completion_ids, responses_text, rewards, full_mask, advantages, loss

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ascend MiniMind PPO")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='ppo_actor', type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-7)
    parser.add_argument("--critic_learning_rate", type=float, default=5e-7)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--max_seq_len', default=768, type=int)
    parser.add_argument("--max_gen_len", type=int, default=1024)
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl")
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--kl_coef", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--cliprange_value", type=float, default=0.2)
    parser.add_argument("--ppo_update_iters", type=int, default=2)
    parser.add_argument("--early_stop_kl", type=float, default=0.25)
    parser.add_argument("--mini_batch_size", type=int, default=2)
    parser.add_argument('--from_weight', default='full_sft', type=str)
    parser.add_argument("--reward_model_path", type=str, default="", help="Reward model path")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Ascend-MiniMind-PPO")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    parser.add_argument("--thinking_ratio", type=float, default=0.9)
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"])
    args = parser.parse_args()

    if args.device is None:
        dev = get_device()
        args.device = f"{dev}:0" if device_count() > 0 else "cpu"

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        dev = get_device()
        args.device = f"{dev}:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    device_type = get_device()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else get_amp_context(dtype=dtype, device_type=device_type)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        wandb.init(project=args.wandb_project, id=wandb_id, resume='must' if wandb_id else None)

    actor_model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    state_dict = torch.load(f'{args.save_dir}/{args.from_weight}_{lm_config.hidden_size}.pth', map_location=args.device)
    critic_model = CriticModel(lm_config).to(args.device)
    critic_model.load_state_dict(state_dict, strict=False)

    rollout_engine = create_rollout_engine(engine_type=args.rollout_engine, policy_model=actor_model, tokenizer=tokenizer, device=args.device, autocast_ctx=autocast_ctx)
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=args.max_seq_len + args.max_gen_len, thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    total_steps = math.ceil(len(train_ds) / args.batch_size / args.accumulation_steps) * args.epochs * args.ppo_update_iters
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_steps, eta_min=args.critic_learning_rate / 10)

    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    if args.use_compile == 1:
        actor_model = torch.compile(actor_model)
        rollout_engine.update_policy(actor_model)
    if dist.is_initialized():
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
    rollout_engine.update_policy(actor_model)

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=(device_type == "cuda"))

        kwargs = {'ref_model': ref_model, 'actor_scheduler': actor_scheduler, 'critic_scheduler': critic_scheduler, 'reward_model': reward_model}
        ppo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, **kwargs, start_step=start_step, wandb=wandb)
