from typing import Callable, Literal

import torch


def compute_group_normalized_reward(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    rollout_batch_size = len(rollout_responses)
    assert rollout_batch_size % group_size == 0
    n_prompts_per_rollout_batch = rollout_batch_size // group_size

    raw_rewards = torch.zeros(rollout_batch_size, dtype=torch.float32)
    for i, rollout_response in enumerate(rollout_responses):
        ground_truth = repeated_ground_truths[i]
        raw_rewards[i] = reward_fn(rollout_response, ground_truth)["reward"]

    grouped = raw_rewards.view(n_prompts_per_rollout_batch, group_size)
    group_mean = grouped.mean(dim=1, keepdim=True)
    advantages = grouped - group_mean
    if normalize_by_std:
        group_std = grouped.std(dim=1, keepdim=True)
        advantages = advantages / (group_std + advantage_eps)
    advantages = advantages.view(-1)

    metadata: dict[str, float] = {
        "reward_mean": raw_rewards.mean().item(),
        "reward_std": raw_rewards.std().item(),
        "reward_max": raw_rewards.max().item(),
        "reward_min": raw_rewards.min().item(),
        "group_reward_mean_mean": group_mean.mean().item(),
        "group_reward_mean_std": group_mean.std().item(),
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std().item(),
        "advantage_max": advantages.max().item(),
        "advantage_min": advantages.min().item(),
    }
    return (advantages, raw_rewards, metadata)


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs:torch.Tensor,
) -> torch.Tensor:
    return torch.neg(raw_rewards_or_advantages * policy_log_probs)


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    unclipped_obj = ratio * advantages
    clipped_obj = clipped_ratio * advantages
    loss = -torch.min(unclipped_obj, clipped_obj)
    is_clipped = (clipped_obj < unclipped_obj)
    metadata: dict[str, torch.Tensor] = {"is_clipped": is_clipped}
    return (loss, metadata)


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
)-> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    match loss_type:
        case "no_baseline":
            loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        case "reinforce_with_baseline":
            loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        case "grpo_clip":
            loss, _ = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    metadata: dict[str, torch.Tensor] = {}
    return (loss, metadata)

