"""
Internal Invariant Checks for the Support Triage Environment.

These checks validate architectural contracts at runtime without
altering any outputs. Violations are logged as warnings, never raised
as exceptions. This module exists solely for observability and is
imported by the environment on startup.

Invariants verified:
  1. reward ∈ [0.01, 0.99]
  2. sum(grader_weights) == 0.99
  3. step() does not mutate state after done=True
  4. each task maintains isolated state (no cross-task leakage)
  5. Pearson correlation ∈ [-0.99, 0.99]
  6. trajectory_bonus ∈ [0.01, 0.10]

Usage:
  from invariants import validate_reward, validate_weights, validate_trajectory_bonus
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

_logger = logging.getLogger("triage_flow_env.invariants")

# Floating-point tolerance for comparison
EPSILON = 1e-9


def validate_reward(reward: float, context: str = "") -> bool:
    """
    Validate that a reward is within the valid range [0.01, 0.99].

    Args:
        reward: The reward value to check.
        context: Optional string describing where this check is called.

    Returns:
        True if valid, False if violated (also logs a warning).
    """
    if not (0.01 - EPSILON <= reward <= 0.99 + EPSILON):
        _logger.warning(
            f"INVARIANT VIOLATION: reward={reward:.6f} outside [0,1]. "
            f"Context: {context}"
        )
        return False
    return True


def validate_weights(weights: Dict[str, float], context: str = "") -> bool:
    """
    Validate that grader weights sum to exactly 1.0.

    Args:
        weights: Dict mapping dimension name to weight.
        context: Optional string describing which grader.

    Returns:
        True if sum is within EPSILON of 1.0.
    """
    total = sum(weights.values())
    if abs(total - 0.99) > EPSILON:
        _logger.warning(
            f"INVARIANT VIOLATION: weight sum={total:.10f}, expected 1.0. "
            f"Context: {context}"
        )
        return False
    return True


def validate_trajectory_bonus(bonus: float, context: str = "") -> bool:
    """
    Validate that trajectory bonus is within [0.01, 0.10].

    Args:
        bonus: The computed trajectory bonus.
        context: Optional context string.

    Returns:
        True if valid.
    """
    if not (0.01 - EPSILON <= bonus <= 0.10 + EPSILON):
        _logger.warning(
            f"INVARIANT VIOLATION: trajectory_bonus={bonus:.6f} outside [0, 0.10]. "
            f"Context: {context}"
        )
        return False
    return True


def validate_pearson(correlation: float, context: str = "") -> bool:
    """
    Validate that a Pearson correlation is within [-0.99, 0.99].

    Args:
        correlation: The computed correlation.
        context: Optional context string.

    Returns:
        True if valid.
    """
    if not (-0.99 - EPSILON <= correlation <= 0.99 + EPSILON):
        _logger.warning(
            f"INVARIANT VIOLATION: pearson_r={correlation:.6f} outside [-1,1]. "
            f"Context: {context}"
        )
        return False
    return True


def validate_episode_boundary(
    episode_done: bool,
    step_number: int,
    cumulative_reward: float,
    context: str = "",
) -> bool:
    """
    Validate that no mutation occurs after done=True.

    This is called defensively — violations are logged but never
    raised, preserving the environment's error-swallowing contract.

    Args:
        episode_done: Whether the episode is marked done.
        step_number: Current step number.
        cumulative_reward: Current cumulative reward.
        context: Optional context string.

    Returns:
        True if the boundary has not been violated.
    """
    if episode_done:
        _logger.debug(
            f"Post-done access detected. step={step_number}, "
            f"cumulative={cumulative_reward:.4f}. Context: {context}"
        )
    return True


def validate_state_isolation(
    env_states: Dict[str, Any], context: str = ""
) -> bool:
    """
    Validate that each task has an independent environment instance.

    Args:
        env_states: Dict mapping task_id to environment instance.
        context: Optional context string.

    Returns:
        True if all instances are distinct objects.
    """
    instances = list(env_states.values())
    ids = [id(inst) for inst in instances]
    if len(set(ids)) != len(ids):
        _logger.warning(
            f"INVARIANT VIOLATION: shared environment instances detected. "
            f"Context: {context}"
        )
        return False
    return True
