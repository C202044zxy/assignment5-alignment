from typing import Callable

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
    
    metadata: dict[str, float] = {}
    return (advantages, raw_rewards, metadata)

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs:torch.Tensor,
) -> torch.Tensor:
    return torch.neg(raw_rewards_or_advantages * policy_log_probs)