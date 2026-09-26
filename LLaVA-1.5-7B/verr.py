import os
import json
import re
import random
import torch
from PIL import Image
from datasets import Dataset
from transformers import LlavaForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer

from utils import *


model_path = "..."
output_dir = "..."
data_dir = "/datasets/..."


model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    attn_implementation="eager"
)


train_dataset = load_train(data_dir)


def identify_visual_heads(attentions, response_positions, prompt_ids, top_k_heads=0.2):
    L = len(attentions)
    H = attentions[0].shape[1]

    visual_propensity = torch.zeros(L, H, device=attentions[0].device)
    vision_token_start, vision_token_end = get_vision_token_range(prompt_ids)

    for l in range(L):
        attn_map = attentions[l][0]
        score = attn_map[:, response_positions, vision_token_start:vision_token_end].sum(dim=-1).mean(dim=-1)
        visual_propensity[l] = score

    flat_scores = visual_propensity.view(-1)
    k = int(L * H * top_k_heads)
    threshold = torch.topk(flat_scores, k).values[-1]
    visual_heads_mask = visual_propensity >= threshold
    visual_heads_indices = torch.nonzero(visual_heads_mask).tolist()

    return visual_heads_indices


def compute_gamma(attentions, response_positions, prompt_ids, gamma_amp=2, top_k_ratio=0.05, window_size=10):
    device = attentions[0].device

    vision_token_start, vision_token_end = get_vision_token_range(prompt_ids)
    num_vis_tokens = 576

    vis_heads = identify_visual_heads(attentions, response_positions, prompt_ids)
    if not vis_heads:
        return torch.ones(len(response_positions), device=device)

    A_vis = torch.zeros((attentions[0].shape[-1], attentions[0].shape[-1]), device=device)
    for l, h in vis_heads:
        A_vis += attentions[l][0, h]
    A_vis /= len(vis_heads)

    prompt_text_indices = list(range(0, vision_token_start)) + list(range(vision_token_end, len(prompt_ids[0])))
    prompt_to_img_attn = A_vis[prompt_text_indices, vision_token_start:vision_token_end]
    
    anchor_scores = prompt_to_img_attn.sum(dim=0)
    
    k = max(int(num_vis_tokens * top_k_ratio), 1)
    threshold_score = torch.topk(anchor_scores, k).values[-1]
    
    anchor_mask = (anchor_scores >= threshold_score).float()

    resp_to_img_attn = A_vis[response_positions, vision_token_start:vision_token_end]
    
    focused_attention = (resp_to_img_attn * anchor_mask).sum(dim=1)
    
    seq_len = focused_attention.shape[0]
    
    if seq_len < window_size:
        local_mean = focused_attention.mean()
        local_std = focused_attention.std() + 1e-6
        z_scores = (focused_attention - local_mean) / local_std
    else:
        pad_len = window_size // 2
        padded_attn = torch.nn.functional.pad(
            focused_attention.unsqueeze(0).unsqueeze(0), 
            (pad_len, pad_len), 
            mode='reflect'
        ).squeeze()
    
        windows = padded_attn.unfold(0, window_size, 1)[:seq_len]
        
        local_mean = windows.mean(dim=1)
        local_std = windows.std(dim=1) + 1e-6
        
        z_scores = (focused_attention - local_mean) / local_std
    

    is_looking_at_key_content = (z_scores > 1.0) & (focused_attention > 1e-3)

    gamma = torch.ones(len(response_positions), device=device)
    if is_looking_at_key_content.any():
        gamma[is_looking_at_key_content] = gamma_amp
        
    gamma = gamma / gamma.mean()
    return gamma.detach()


class VERRTrainer(GRPOTrainer):
    def __init__(self, *args, loss_type="grpo", **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type

    def _generate_and_score_completions(self, inputs):
        outputs = super()._generate_and_score_completions(inputs)

        device = self.accelerator.device
        batch_size = outputs["completion_ids"].shape[0]

        pad_token_id = 32001

        gamma_t_list = []

        for i in range(batch_size):
            prompt_ids = outputs["prompt_ids"][i:i + 1]
            completion_ids = outputs["completion_ids"][i:i + 1]
            full_ids = torch.cat([prompt_ids, completion_ids], dim=1)

            attention_mask = (full_ids != pad_token_id).long()

            extra_kwargs = {}
            if "pixel_values" in outputs:
                extra_kwargs["pixel_values"] = outputs["pixel_values"][i:i + 1]
            if "image_sizes" in outputs:
                extra_kwargs["image_sizes"] = outputs["image_sizes"][i:i + 1]
            if "image_grid_thw" in outputs:
                extra_kwargs["image_grid_thw"] = outputs["image_grid_thw"][i:i + 1]
            if "pixel_attention_mask" in outputs:
                extra_kwargs["pixel_attention_mask"] = outputs["pixel_attention_mask"][i:i + 1]

            with torch.no_grad():
                attn_outputs = self.model(
                    input_ids=full_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    output_attentions=True,
                    return_dict=True,
                    **extra_kwargs
                )

            prompt_physical_len = prompt_ids.shape[1]

            valid_completion_len = (completion_ids != pad_token_id).sum().item()

            response_positions = list(range(prompt_physical_len, prompt_physical_len + valid_completion_len))

            gamma_t = compute_gamma(
                attentions=attn_outputs.attentions,
                prompt_ids=prompt_ids,
                response_positions=response_positions
            )

            current_physical_len = completion_ids.shape[1]

            if gamma_t.shape[0] < current_physical_len:
                pad_len = current_physical_len - gamma_t.shape[0]
                gamma_t = torch.cat([
                    gamma_t,
                    torch.ones(pad_len, device=gamma_t.device)
                ])

            gamma_t = gamma_t[:current_physical_len]

            gamma_t_list.append(gamma_t)

            del attn_outputs
            torch.cuda.empty_cache()

        gamma_t_batch = torch.stack(gamma_t_list)  # [B, C]
        outputs["gamma_t"] = gamma_t_batch

        return outputs

    def _compute_loss(self, model, inputs):
        advantages = inputs["advantages"]  # [B]
        gamma_t = inputs["gamma_t"]  # [B, C]

        token_advantages = advantages.unsqueeze(1) * gamma_t  # [B, C]

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(
                entropies, completion_mask, 1 - self.top_entropy_quantile
            )
        else:
            entropy_mask = None

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                    torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        log_ratio = per_token_logps - old_per_token_logps

        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown importance sampling level: {self.importance_sampling_level}")

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * token_advantages
        per_token_loss2 = coef_2 * token_advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        mode = "train" if self.model.training else "eval"
        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (token_advantages < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (token_advantages > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        return loss


training_args = GRPOConfig(
    output_dir=output_dir,
    num_generations=4,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=5e-6,
    max_prompt_length=None,
    max_completion_length=512,
    temperature=1.0,
    bf16=True,
    tf32=True
)

trainer = VERRTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    reward_funcs=[sqa_reward_func, reason_reward_func]
)
trainer.train()
trainer.save_model(output_dir)
