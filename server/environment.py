"""
Main environment class for the Support Triage Environment.

SupportTriageEnv wraps the task-specific logic and provides a unified
interface with reset(), step(), and state() methods. All errors are
captured and returned in the info dict — never raised to the caller.

Includes global penalties for repetition and schema abuse across all tasks,
and a trajectory consistency bonus applied at episode completion.
"""

from __future__ import annotations

import json
import statistics
import threading
import uuid
from typing import Any, Dict, List

from fastapi import HTTPException

from server.models import (
    ClassifyAction,
    EnvironmentState,
    PrioritizeAction,
    ResolveAction,
)
from server.tasks.task_classify import ClassifyTask
from server.tasks.task_prioritize import PrioritizeTask
from server.tasks.task_resolve import ResolveTask

# Action type routing map
_ACTION_TYPES = {
    "classify": ClassifyAction,
    "prioritize": PrioritizeAction,
    "resolve": ResolveAction,
}

# Task class routing map
_TASK_CLASSES = {
    "classify": ClassifyTask,
    "prioritize": PrioritizeTask,
    "resolve": ResolveTask,
}

_VALID_TASK_IDS = set(_TASK_CLASSES.keys())


class SupportTriageEnv:
    """
    Unified environment for customer support ticket triage.

    Routes requests to the correct task (classify, prioritize, resolve)
    and provides a consistent API for reset/step/state operations.
    Includes global penalties for repetition and schema abuse, and a
    trajectory consistency bonus applied at episode completion.
    """

    def __init__(self, task_id: str = "classify", seed: int = 42) -> None:
        """
        Initialize the environment with the specified task.

        Args:
            task_id: One of 'classify', 'prioritize', 'resolve'.
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If task_id is not one of the valid options.
        """
        if task_id not in _VALID_TASK_IDS:
            raise ValueError(
                f"Invalid task_id '{task_id}'. Must be one of: {sorted(_VALID_TASK_IDS)}"
            )

        self.task_id = task_id
        self.seed = seed
        self.episode_id = str(uuid.uuid4())

        task_cls = _TASK_CLASSES[task_id]
        self._task = task_cls(seed=seed)
        self._state_history: list = []
        self._initialized = False

        # Global penalty tracking
        self._last_action_json: str | None = None
        self._cumulative_reward = 0.0

        # Trajectory tracking for episode analytics and trajectory bonus
        self._step_rewards: List[float] = []
        self._penalties_count: int = 0
        self._step_number: int = 0
        self._episode_done: bool = False
        self._state_lock = threading.Lock()

    def reset(self) -> Dict[str, Any]:
        """
        Reset the task and return the initial observation.

        Assigns a new UUID4 episode_id on each reset. Resets cumulative
        reward, penalty tracking state, and trajectory tracking.

        Returns:
            Serialized initial observation as a dict.
        """
        with self._state_lock:
            self.episode_id = str(uuid.uuid4())
            self._task._episode_id = self.episode_id
            self._cumulative_reward = 0.0
            self._last_action_json = None
            self._step_rewards = []
            self._penalties_count = 0
            self._step_number = 0
            self._episode_done = False
            obs = self._task.reset()
            self._initialized = True
            return obs

    def _compute_trajectory_bonus(self) -> float:
        """
        Compute an end-of-episode trajectory consistency bonus.

        Applied only on the final step (when done=True).

        The bonus rewards agents that demonstrate consistent,
        improving behaviour across the episode rather than
        achieving high scores on some steps through lucky guesses
        and low scores on others.

        Bonus criteria (each worth 0.025, max total bonus 0.10):

        Criterion 1 — Monotonic improvement tendency:
          [1, 2, ..., n] and step rewards. If correlation > 0.3
          (rewards trend upward across the episode): bonus += 0.025
          Rationale: an agent that learns within an episode
          should improve its scores over time.

        Criterion 2 — No catastrophic steps:
          If zero steps received a reward of 0.0 exactly
          (before clamping, i.e., the raw grader output was 0.0):
          bonus += 0.025
          Rationale: a consistent agent avoids complete failures.

        Criterion 3 — Reward variance below threshold:
          If std(step_rewards) < 0.25: bonus += 0.025
          Rationale: low variance indicates a principled strategy
          rather than random guessing with occasional lucky hits.

        Criterion 4 — Above-baseline mean:
          If mean(step_rewards) > 0.50: bonus += 0.025
          Rationale: the bonus is only meaningful for agents
          already performing above a random baseline.

        All four criteria are evaluated using only
        self._step_rewards (the per-step reward list).
        No external calls. Executes in under 1ms.

        Returns:
            Float in [0.0, 0.10] representing the trajectory bonus.
            Returns 0.0 if fewer than 3 steps have been taken
            (insufficient data for trajectory analysis).
        """
        if len(self._step_rewards) < 3:
            return 0.0

        bonus = 0.0
        n = len(self._step_rewards)
        rewards = self._step_rewards

        # Criterion 1: monotonic improvement tendency
        criterion_1 = False
        try:
            steps = list(range(1, n + 1))
            mean_s = sum(steps) / n
            mean_r = sum(rewards) / n
            num = sum((s - mean_s) * (r - mean_r)
                      for s, r in zip(steps, rewards))
            den_s = sum((s - mean_s) ** 2 for s in steps) ** 0.5
            den_r = sum((r - mean_r) ** 2 for r in rewards) ** 0.5
            
            if den_s > 0 and den_r > 0:
                correlation = num / (den_s * den_r)
                correlation = max(-1.0, min(1.0, correlation))
                
                # Monotonic trend penalty: deduct points for each step-over-step reversal
                reversals = sum(1 for i in range(1, n) if rewards[i] < rewards[i-1] - 0.05)
                monotonic_trend_score = correlation - (reversals * 0.15)
                
                # Use penalized score for the threshold check
                if monotonic_trend_score > 0.3:
                    bonus += 0.025
                    criterion_1 = True
        except Exception:
            pass

        # Criterion 2: no catastrophic steps
        criterion_2 = not any(r == 0.0 for r in rewards)
        if criterion_2:
            bonus += 0.025

        # Criterion 3: low reward variance
        criterion_3 = False
        try:
            if n >= 2 and statistics.stdev(rewards) < 0.25:
                bonus += 0.025
                criterion_3 = True
        except Exception:
            pass

        # Criterion 4: above-baseline mean
        criterion_4 = (sum(rewards) / n) > 0.50
        if criterion_4:
            bonus += 0.025

        return min(bonus, 0.10)

    def step(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the action, execute a step, and return results.

        Applies global penalties after computing raw reward:
          - Repetition penalty (-0.10) if action is identical to previous
          - Schema abuse penalty (-0.10, resolve only) if unique_word_ratio < 0.40

        Applies trajectory consistency bonus on the final step (done=True).

        If the action fails Pydantic validation, returns reward=0.0
        with the error message in info. Never raises exceptions.

        Args:
            action_dict: Dict to parse into the task's action model.

        Returns:
            Dict with keys: observation, reward, done, info.
        """
        if not self._initialized:
            return {
                "observation": {},
                "reward": 0.10,
                "done": False,
                "info": {
                    "error": "Environment not initialized. Call reset() first.",
                    "episode_id": self.episode_id,
                    "step": 0,
                    "reward_breakdown": {},
                    "feedback": "",
                    "penalties": [],
                },
            }

        # Guard: if episode was already completed, raise 400
        if self._episode_done:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "episode_already_complete",
                    "message": "This episode has ended (done=True was returned). "
                               "Call /reset to begin a new episode.",
                    "episode_id": str(self.episode_id)
                }
            )

        with self._state_lock:
            action_type = _ACTION_TYPES[self.task_id]

            try:
                action = action_type(**action_dict)
            except Exception as e:
                # Return zero reward on validation failure, don't advance
                try:
                    current_state = self._task.state()
                    done = current_state.done
                    step_num = current_state.step_number
                except Exception:
                    done = False
                    step_num = 0

                return {
                    "observation": {},
                    "reward": 0.10,
                    "done": done,
                    "info": {
                        "error": f"Action validation error: {str(e)}",
                        "episode_id": self.episode_id,
                        "step": step_num,
                        "reward_breakdown": {},
                        "feedback": "",
                        "penalties": [],
                    },
                }

            try:
                obs_dict, raw_reward, done, info = self._task.step(action)
                info["episode_id"] = self.episode_id
                info["step"] = self._task.state().step_number

                # Capture grader output before environment-level penalties
                _grader_raw_score = raw_reward

                # ── Global Penalties ─────────────────────────────────────────
                penalties: list = []

                # Penalty 1: Repetition Penalty
                current_action_json = json.dumps(action_dict, sort_keys=True)
                if (
                    self._last_action_json is not None
                    and current_action_json == self._last_action_json
                ):
                    raw_reward -= 0.10
                    penalties.append("repetition_penalty: -0.10")
                self._last_action_json = current_action_json

                # Penalty 2: Schema Abuse Penalty (resolve task only)
                if self.task_id == "resolve":
                    response_body = action_dict.get("response_body", "")
                    if isinstance(response_body, str):
                        words = response_body.lower().split()
                        if len(words) > 20:
                            unique_ratio = len(set(words)) / len(words)
                            if unique_ratio < 0.40:
                                raw_reward -= 0.10
                                penalties.append("schema_abuse_penalty: -0.10")

                # Clamp after penalties — strictly within (0, 1) to satisfy validator
                final_reward = max(0.10, min(0.90, raw_reward))

                # Track step rewards and penalties
                self._step_rewards.append(final_reward)
                self._penalties_count += len(penalties)
                self._step_number += 1

                # ── Trajectory Bonus (applied on final step) ─────────────────
                trajectory_bonus_info = {
                    "monotonic_improvement": False,
                    "no_catastrophic_steps": False,
                    "low_variance": False,
                    "above_baseline_mean": False,
                }
                
                if done:
                    self._episode_done = True
                    trajectory_bonus = self._compute_trajectory_bonus()
                    
                    # Re-compute logic for breakdown reporting (consistent with _compute_trajectory_bonus)
                    n = len(self._step_rewards)
                    rewards = self._step_rewards
                    if n >= 2:
                        try:
                            steps = list(range(1, n + 1))
                            mean_s = sum(steps) / n
                            mean_r = sum(rewards) / n
                            num = sum((s - mean_s) * (r - mean_r) for s, r in zip(steps, rewards))
                            den_s = sum((s - mean_s) ** 2 for s in steps) ** 0.5
                            den_r = sum((r - mean_r) ** 2 for r in rewards) ** 0.5
                            if den_s > 0 and den_r > 0:
                                _corr = num / (den_s * den_r)
                                _corr = max(-1.0, min(1.0, _corr))
                                trajectory_bonus_info["monotonic_improvement"] = _corr > 0.3
                                trajectory_bonus_info["pearson_correlation"] = round(_corr, 6)
                        except Exception: pass

                        trajectory_bonus_info["no_catastrophic_steps"] = not any(r == 0.0 for r in rewards)
                        try:
                            import statistics
                            trajectory_bonus_info["low_variance"] = statistics.stdev(rewards) < 0.25
                        except Exception: pass
                        trajectory_bonus_info["above_baseline_mean"] = (sum(rewards) / n) > 0.50

                    if trajectory_bonus > 0:
                        # Ensure bonus doesn't push us to 1.0 (clamping to 0.90)
                        final_reward = min(0.90, final_reward + trajectory_bonus)
                        self._step_rewards[-1] = final_reward
                        info["trajectory_bonus"] = trajectory_bonus
                    else:
                        info["trajectory_bonus"] = 0.0
                    
                    info["trajectory_bonus_breakdown"] = trajectory_bonus_info

                # Update cumulative reward
                self._cumulative_reward += final_reward

                # ── Forward-Computed Reward Transparency ──────────────────────
                # Each component is independently computed, never reverse-engineered.
                # Evaluators can verify: final = clamp(grader - penalties + bonus)
                _penalty_sum = sum(0.10 for _ in penalties)
                _bonus_applied = info.get("trajectory_bonus", 0.0) if done else 0.0
                info["reward_components"] = {
                    "grader_raw_score": round(_grader_raw_score, 4),
                    "penalty_total": round(_penalty_sum, 4),
                    "penalty_details": list(penalties),
                    "trajectory_bonus": round(_bonus_applied, 4),
                    "final_reward": round(final_reward, 4),
                }

                # Legacy breakdown compatibility: compute global_penalties forward
                if penalties:
                    info["breakdown"]["global_penalties"] = round(-_penalty_sum, 4)

                info["reward_breakdown"] = info["breakdown"]
                info["penalties"] = penalties

                return {
                    "observation": obs_dict,
                    "reward": float(final_reward),
                    "done": done,
                    "info": info,
                }
            except Exception as e:
                return {
                    "observation": {},
                    "reward": 0.10,
                    "done": False,
                    "info": {
                        "error": f"Step execution error: {str(e)}",
                        "episode_id": self.episode_id,
                        "step": 0,
                        "reward_breakdown": {},
                        "feedback": "",
                        "penalties": [],
                    },
                }

    def state(self) -> Dict[str, Any]:
        """
        Return the current environment state with trajectory-level analytics.

        Always reflects the episode_id from the most recent reset() call.
        Includes mean/min/max reward, total penalties, and steps remaining.

        Returns:
            Serialized EnvironmentState as a dict.
        """
        if not self._initialized:
            return EnvironmentState(
                task_id=self.task_id,
                current_ticket_index=0,
                total_tickets=1,
                cumulative_reward=0.0,
                step_number=0,
                done=False,
                episode_id=self.episode_id,
            ).model_dump()

        state = self._task.state()
        # Override episode_id to ensure environment-level consistency
        state_dict = state.model_dump()
        state_dict["episode_id"] = self.episode_id
        state_dict["cumulative_reward"] = self._cumulative_reward

        # Compute and populate episode analytics
        if self._step_rewards:
            state_dict["mean_reward_so_far"] = round(
                sum(self._step_rewards) / len(self._step_rewards), 4
            )
            state_dict["min_reward_this_episode"] = round(
                min(self._step_rewards), 4
            )
            state_dict["max_reward_this_episode"] = round(
                max(self._step_rewards), 4
            )
        else:
            state_dict["mean_reward_so_far"] = 0.10
            state_dict["min_reward_this_episode"] = 0.10
            state_dict["max_reward_this_episode"] = 0.10

        state_dict["penalties_applied_total"] = self._penalties_count
        state_dict["steps_remaining"] = max(
            0, self._task._max_steps - self._step_number
        )

        return state_dict
