"""
Reward utilities for Weird AI.

This module connects Chapter 6 of Build a Reasoning Model from Scratch to the
Weird AI project.

The book uses RLVR: reinforcement learning with verifiable rewards.

For math:
    generated answer -> deterministic verifier -> reward 0 or 1

For Weird AI:
    generated lyrics -> project evaluator -> heuristic reward between 0 and 1

This file does not train the model yet. Instead, it prepares the reward and
group-relative advantage data that a future GRPO-style training loop would need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Callable, Iterable, Optional


EvaluationFunction = Callable[[str], dict]


@dataclass
class RolloutReward:
    """
    Store reward information for one generated rollout.

    Attributes:
        prompt: The original prompt.
        text: The generated lyrics.
        evaluation: Full evaluation dictionary for the rollout.
        reward: Normalized reward value.
        advantage: Group-relative advantage.
        index: Rollout number, starting at 1.
    """

    prompt: str
    text: str
    evaluation: dict
    reward: float
    advantage: float
    index: int


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """
    Clamp a value to a numeric range.

    Args:
        value: Numeric value to clamp.
        minimum: Lowest allowed value.
        maximum: Highest allowed value.

    Returns:
        The clamped value.
    """

    # TODO:
    # 1. Return minimum if value is below minimum.
    # 2. Return maximum if value is above maximum.
    # 3. Otherwise return value.

    if value < minimum:
        return minimum
    elif value > maximum:
        return maximum
    return value


def extract_score(evaluation: dict, score_key: str = "overall_score") -> float:
    """
    Extract a numeric score from an evaluation dictionary.

    Args:
        evaluation: Evaluation dictionary, usually from weird_ai.evaluate.
        score_key: Key containing the main score.

    Returns:
        Score as a float. Missing or invalid values become 0.0.
    """

    # TODO:
    # 1. Read score_key from evaluation.
    # 2. Convert it to float.
    # 3. Return 0.0 if the key is missing or invalid.

    try:
        return float(evaluation.get(score_key, 0.0))
    except (ValueError, TypeError):
        return 0.0


def normalize_reward(
    score: float,
    *,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """
    Normalize a score into the 0.0 to 1.0 reward range.

    Args:
        score: Raw score.
        minimum: Expected minimum score.
        maximum: Expected maximum score.

    Returns:
        Normalized reward in the range [0.0, 1.0].
    """

    # TODO:
    # 1. Handle the case where maximum equals minimum.
    if maximum == minimum:
            return 0.0
    
    # 2. Convert score to the 0.0 to 1.0 range.
    normalized = (score - minimum) / (maximum - minimum)

    # 3. Clamp the result to [0.0, 1.0].
    return clamp(normalized)


def reward_from_evaluation(
    evaluation: dict,
    *,
    score_key: str = "overall_score",
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """
    Convert an evaluation dictionary into a reward.

    Args:
        evaluation: Evaluation dictionary.
        score_key: Key containing the main score.
        minimum: Expected minimum score.
        maximum: Expected maximum score.

    Returns:
        Normalized reward.
    """

    # TODO:
    # 1. Extract the score from evaluation.
    score = extract_score(evaluation, score_key)

    # 2. Normalize the score into a reward.
    # 3. Return the reward.
    return normalize_reward(
        score,
        minimum=minimum,
        maximum=maximum
    )


def compute_group_advantages(
    rewards: Iterable[float],
    *,
    normalize: bool = False,
    epsilon: float = 1e-8,
) -> list[float]:
    """
    Compute group-relative advantages.

    In GRPO-style training, each rollout is compared with the other rollouts
    generated for the same prompt.

    Basic advantage:
        advantage = reward - mean(group_rewards)

    Optional normalized advantage:
        advantage = (reward - mean(group_rewards)) / std(group_rewards)

    Args:
        rewards: Reward values for a group of rollouts.
        normalize: Whether to divide by group standard deviation.
        epsilon: Small value to avoid division by zero.

    Returns:
        List of advantages in the same order as rewards.
    """

    # TODO:
    # 1. Convert rewards to a list.
    rewards = list(rewards)

    # 2. Return an empty list for an empty input.
    if not rewards:
            return []
    
    # 3. Compute the group mean.
    group_mean = mean(rewards)

    # 4. Subtract the mean from each reward.
    advtantages = [
        reward - group_mean for reward in rewards
    ]

    # 5. If normalize=True, divide each advantage by the group standard deviation.
    if normalize:
        group_deviation = pstdev(rewards)

        advtantages = [
            advantage / (group_deviation + epsilon) for advantage in advtantages
        ]

    return advtantages


def build_rollout_reward(
    prompt: str,
    text: str,
    evaluation: dict,
    reward: float,
    advantage: float,
    index: int,
) -> RolloutReward:
    """
    Build a RolloutReward object.

    Args:
        prompt: Original prompt.
        text: Generated lyrics.
        evaluation: Evaluation dictionary.
        reward: Reward value.
        advantage: Group-relative advantage.
        index: Rollout number.

    Returns:
        RolloutReward instance.
    """

    # TODO:
    # 1. Return a RolloutReward using the provided values.
    return RolloutReward(
        prompt=prompt,
        text=text,
        evaluation=evaluation,
        reward=reward,
        advantage= advantage,
        index=index
    )


def prepare_reward_batch(
    prompt: str,
    rollouts: list[str],
    evaluation_func: EvaluationFunction,
    *,
    score_key: str = "overall_score",
    reward_minimum: float = 0.0,
    reward_maximum: float = 1.0,
    normalize_advantages: bool = False,
) -> list[RolloutReward]:
    """
    Evaluate rollouts, compute rewards, and compute group-relative advantages.

    This is the main assignment function. It prepares the information a future
    GRPO-style training loop would need.

    Args:
        prompt: Original prompt.
        rollouts: Generated lyric candidates for the same prompt.
        evaluation_func: Function that evaluates one rollout.
        score_key: Key containing the main score in the evaluation dictionary.
        reward_minimum: Expected minimum evaluation score.
        reward_maximum: Expected maximum evaluation score.
        normalize_advantages: Whether to standardize advantages.

    Returns:
        List of RolloutReward objects.
    """

    # TODO:
    # 1. Evaluate each rollout with evaluation_func.
    if not rollouts:
        return []

    evaluations = [
        evaluation_func(text) for text in rollouts
    ]

    # 2. Convert each evaluation to a reward.
    rewards = [
        reward_from_evaluation(
            evaluation,
            score_key=score_key,
            minimum=reward_minimum,
            maximum=reward_maximum
        )
        for evaluation in evaluations
    ]

    # 3. Compute group-relative advantages from the rewards.
    advantages = compute_group_advantages(
        rewards,
        normalize=normalize_advantages
    )

    # 4. Build and return one RolloutReward per rollout.
    # 5. Rollout indexes should start at 1.
    _rollouts = []

    for index, (text, evaluation, reward, advantage) in enumerate(
        zip(rollouts, evaluations, rewards, advantages),
        start=1
    ):
        _rollouts.append(
            build_rollout_reward(
                prompt=prompt,
                text=text,
                evaluation=evaluation,
                reward=reward,
                advantage=advantage,
                index=index
            )
        )

    return _rollouts


def rank_by_reward(batch: list[RolloutReward], *, descending: bool = True) -> list[RolloutReward]:
    """
    Sort rollout records by reward.

    Args:
        batch: RolloutReward objects.
        descending: If True, highest reward first.

    Returns:
        Sorted RolloutReward list.
    """

    # TODO:
    # 1. Sort the batch by reward.
    # 2. Return the sorted list.
    #
    # Do not mutate the original list.
    return sorted(
        batch,
        key=lambda rollout: rollout.reward,
        reverse=descending
    )


def get_best_rollout(batch: list[RolloutReward]) -> Optional[RolloutReward]:
    """
    Return the highest-reward rollout.

    Args:
        batch: RolloutReward objects.

    Returns:
        Best rollout, or None if batch is empty.
    """

    # TODO:
    # 1. Return None for an empty batch.
    # 2. Return the rollout with the highest reward.
    if not batch:
        return None

    return max(batch, key=lambda rollout: rollout.reward)


def get_worst_rollout(batch: list[RolloutReward]) -> Optional[RolloutReward]:
    """
    Return the lowest-reward rollout.

    Args:
        batch: RolloutReward objects.

    Returns:
        Worst rollout, or None if batch is empty.
    """

    # TODO:
    # 1. Return None for an empty batch.
    # 2. Return the rollout with the lowest reward.
    if not batch:
        return None

    return min(batch, key=lambda rollout: rollout.reward)


def summarize_reward_batch(batch: list[RolloutReward]) -> str:
    """
    Create a readable summary of a reward batch.

    Args:
        batch: RolloutReward objects.

    Returns:
        Multi-line summary string.
    """

    # TODO:
    # 1. Handle an empty batch.
    if not batch:
        return "No rollouts in reward batch."

    # 2. Include number of rollouts.
    average_reward = mean(rollout.reward for rollout in batch)

    # 3. Include average reward.
    best = get_best_rollout(batch)
    worst = get_worst_rollout(batch)

    # 4. Include best and worst rollout indexes and rewards.
    # 5. Include each rollout's reward and advantage.
    lines = [
        f"Number of rollouts: {len(batch)}",
        f"Average reward: {average_reward:.4f}",
        f"Best rollout: #{best.index} (Reward: {best.reward:.4f})",
        f"Worst rollout: #{worst.index} (Reward: {worst.reward:.4f})",
        "",
        "Individual rollout results:"
    ]

    for rollout in batch:
        lines.append(
            f"Rollout #{rollout.index}: "
            f"Reward = {rollout.reward:.4f}, "
            f"Advantage = {rollout.advantage:.4f}"
        )

    return "\n".join(lines)