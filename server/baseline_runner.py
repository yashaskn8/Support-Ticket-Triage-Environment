"""
Baseline measurement tool for the Support Triage Environment.

Runs all three tasks (classify -> prioritize -> resolve) against the
environment HTTP API using an LLM, measures scores, writes results to
baseline_scores.json, and optionally auto-updates the README.md table.

This is NOT the judging inference script (that is inference.py).
This script's purpose is to populate baseline_scores.json with verified
scores for documentation and submission validation.

Usage:
    export HF_TOKEN=<your-hf-token>
    export API_BASE_URL=https://router.huggingface.co/v1
    export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
    export ENV_BASE_URL=http://localhost:7860
    python baseline_runner.py
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

from server.llm_utils import (
    RESOLVE_SYSTEM_PROMPT,
    CLASSIFY_SYSTEM_PROMPT,
    PRIORITIZE_SYSTEM_PROMPT,
    build_resolve_user_message,
    normalise_resolve_action,
    parse_llm_json,
)

# ── Configuration ────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME")
if not MODEL_NAME:
    raise ValueError("MODEL_NAME environment variable is required.")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

def parse_args():
    """
    Parse command-line arguments for baseline_runner.py.
    Supports --seed for reproducible runs and 
    --seed random for non-deterministic multi-seed evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Measure baseline scores for support-triage-env."
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="42",
        help=(
            "Random seed for episode determinism. "
            "Use an integer (e.g., --seed 42) for a specific seed, "
            "or 'random' to sample a new seed on each run. "
            "Default: 42."
        ),
    )
    return parser.parse_args()


# ── Task definitions ─────────────────────────────────────────────────────────

TASKS = [
    {
        "task_id": "classify",
        "max_steps": 10,
        "temperature": 0.1,
        "max_tokens": 500,
        "system_prompt": CLASSIFY_SYSTEM_PROMPT,
    },
    {
        "task_id": "prioritize",
        "max_steps": 10,
        "temperature": 0.1,
        "max_tokens": 500,
        "system_prompt": PRIORITIZE_SYSTEM_PROMPT,
    },
    {
        "task_id": "resolve",
        "max_steps": 5,
        "temperature": 0.3,
        "max_tokens": 1200,
        "system_prompt": RESOLVE_SYSTEM_PROMPT,
    },
]


def build_action(task_id: str, llm_output: str, observation: dict) -> dict:
    """
    Parse LLM output into a valid action dict for the given task.

    Args:
        task_id: One of 'classify', 'prioritize', 'resolve'.
        llm_output: Raw string output from the LLM.
        observation: Current observation dict (used for resolve context).

    Returns:
        Parsed and normalised action dict.
    """
    action = parse_llm_json(llm_output)

    if task_id == "resolve" and action:
        action = normalise_resolve_action(action)

    if not action:
        # Fallback defaults
        if task_id == "classify":
            action = {"category": "GENERAL"}
        elif task_id == "prioritize":
            action = {
                "priority": "MEDIUM",
                "assigned_team": "general_team",
                "estimated_resolution_hours": 24,
            }
        else:
            action = {
                "response_subject": "Re: Your Support Request",
                "response_body": (
                    "Dear Customer,\n\nWe apologize for the inconvenience. "
                    "Our team will investigate and resolve this promptly. "
                    "We have already begun looking into your issue.\n\n"
                    "Best regards,\nCustomer Support Team"
                ),
                "internal_notes": "",
                "escalate": False,
            }

    return action


def detect_suspicious_scores(results: dict) -> List[str]:
    """
    Analyse baseline results for suspicious patterns that indicate
    measurement errors rather than genuine LLM performance.

    Args:
        results: The full results dict with 'tasks' key.

    Returns:
        List of warning strings. Empty if no issues detected.
    """
    warnings: List[str] = []
    tasks = results.get("tasks", {})

    for task_id, task_data in tasks.items():
        mean_score = task_data.get("mean_score", 0.0)
        rewards = task_data.get("per_step_rewards", [])

        # Check for perfect scores
        if mean_score >= 0.999:
            warnings.append(
                f"{task_id}: mean_score={mean_score:.3f} is suspiciously "
                f"perfect. Verify grader is not using oracle access."
            )

        # Check for all-identical rewards
        if rewards and len(set(rewards)) == 1:
            warnings.append(
                f"{task_id}: all {len(rewards)} per-step rewards are "
                f"identical ({rewards[0]}). This suggests stubbed data."
            )

        # Check for out-of-range scores
        if mean_score < 0.0 or mean_score > 1.0:
            warnings.append(
                f"{task_id}: mean_score={mean_score:.3f} is outside [0, 1]."
            )

    return warnings


def _run_task(
    task: dict,
    client: OpenAI,
    http_client: httpx.Client,
    resolved_seed: int,
) -> Dict[str, Any]:
    """Run a single task's episode and collect rewards."""
    task_id = task["task_id"]
    max_steps = task["max_steps"]
    temperature = task["temperature"]
    max_tokens = task["max_tokens"]
    system_prompt = task["system_prompt"]

    print(f"\n{'─' * 50}", file=sys.stderr)
    print(f"  Baseline: {task_id} ({max_steps} steps, T={temperature})",
          file=sys.stderr)
    print(f"{'─' * 50}", file=sys.stderr)

    # Reset
    try:
        reset_resp = http_client.post(
            f"{ENV_BASE_URL}/reset",
            json={"task_id": task_id, "seed": resolved_seed},
            timeout=15.0,
        )
        reset_resp.raise_for_status()
        observation = reset_resp.json()
    except Exception as e:
        print(f"  ERROR: Reset failed: {e}", file=sys.stderr)
        return {
            "mean_score": 0.0,
            "steps": 0,
            "per_step_rewards": [],
            "success": False,
        }

    rewards: List[float] = []
    done = False
    step = 0

    while not done and step < max_steps:
        step += 1

        # Build user message
        if task_id == "resolve":
            user_message = build_resolve_user_message(observation)
        else:
            user_message = json.dumps(observation, indent=2, ensure_ascii=False)

        # Call LLM
        action_dict = None
        for attempt in range(3):

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=30.0,
                )
                llm_output = completion.choices[0].message.content or ""
                action_dict = build_action(task_id, llm_output, observation)
                if action_dict:
                    break
            except Exception as e:
                print(f"  Step {step}: LLM error (attempt {attempt+1}): {e}",
                      file=sys.stderr)
                time.sleep(2 ** attempt)

        if action_dict is None:
            action_dict = build_action(task_id, "", observation)

        # Submit action
        try:
            step_resp = http_client.post(
                f"{ENV_BASE_URL}/step",
                json={"task_id": task_id, "action": action_dict},
                timeout=15.0,
            )
            step_resp.raise_for_status()
            result = step_resp.json()
            reward = max(0.01, min(0.99, result.get("reward", 0.01)))
            done = result.get("done", False)
            observation = result.get("observation", {})
            rewards.append(reward)
            print(f"  Step {step}: reward={reward:.3f} done={done}",
                  file=sys.stderr)
        except Exception as e:
            print(f"  Step {step}: Step API error: {e}", file=sys.stderr)
            rewards.append(0.01)

    mean_score = round(sum(rewards) / len(rewards), 4) if rewards else 0.0
    success = mean_score >= 0.40

    print(f"  Result: mean={mean_score:.4f}, steps={step}, success={success}",
          file=sys.stderr)

    return {
        "mean_score": mean_score,
        "steps": step,
        "per_step_rewards": rewards,
        "success": success,
    }


def _get_git_commit() -> str:
    """Get current git commit SHA, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _update_readme(results: dict) -> None:
    """Auto-update the README.md baseline scores table."""
    readme_path = pathlib.Path(__file__).parent / "README.md"
    if not readme_path.exists():
        print("  README.md not found, skipping auto-update", file=sys.stderr)
        return

    content = readme_path.read_text(encoding="utf-8")
    tasks = results.get("tasks", {})

    for task_id, task_data in tasks.items():
        score = task_data["mean_score"]
        steps = task_data["steps"]
        success = "Yes" if task_data["success"] else "No"
        # Match table rows like: | classify | Qwen/... | 0.797 | 10 | Yes |
        pattern = (
            rf"\| {task_id}\s*\|[^|]+\|\s*[\d.]+\s*\|\s*\d+\s*\|\s*\w+\s*\|"
        )
        replacement = (
            f"| {task_id} | {MODEL_NAME} | {score} | {steps} | {success} |"
        )
        content = re.sub(pattern, replacement, content)

    readme_path.write_text(content, encoding="utf-8")
    print("  README.md baseline table updated", file=sys.stderr)


def main() -> None:
    """Run baseline measurements across all three tasks."""
    if not HF_TOKEN:
        print(
            "[ERROR] HF_TOKEN is required. "
            "Set it before running: export HF_TOKEN=<your-token>",
            file=sys.stderr,
        )
        sys.exit(1)

    args = parse_args()
    if args.seed.lower() == "random":
        import random as _random
        resolved_seed = _random.randint(0, 999999)
        print(
            f"[BASELINE] Random seed selected: {resolved_seed}",
            file=sys.stderr,
        )
    else:
        try:
            resolved_seed = int(args.seed)
        except ValueError:
            print(
                f"[ERROR] Invalid seed value: {args.seed!r}. "
                "Use an integer or 'random'.",
                file=sys.stderr,
            )
            sys.exit(1)

    print("=" * 60, file=sys.stderr)
    print("  Support Triage Environment — Baseline Runner", file=sys.stderr)
    print(f"  Model: {MODEL_NAME}", file=sys.stderr)
    print(f"  Environment: {ENV_BASE_URL}", file=sys.stderr)
    print(f"  Seed: {resolved_seed}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )
    http_client = httpx.Client(timeout=15.0)

    # Health check
    for attempt in range(5):
        try:
            resp = http_client.get(f"{ENV_BASE_URL}/health", timeout=5.0)
            if resp.status_code == 200:
                print("  Server ready", file=sys.stderr)
                break
        except Exception:
            pass
        if attempt < 4:
            print(f"  Waiting for server (attempt {attempt+1}/5)...",
                  file=sys.stderr)
            time.sleep(2)
    else:
        print("  ERROR: Server not ready after 5 attempts", file=sys.stderr)
        sys.exit(1)

    # Run all tasks
    task_results = {}
    for task in TASKS:
        task_results[task["task_id"]] = _run_task(task, client, http_client, resolved_seed)

    http_client.close()

    # Build results
    overall_scores = [t["mean_score"] for t in task_results.values()]
    overall_mean = round(sum(overall_scores) / len(overall_scores), 4)

    results = {
        "model": MODEL_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": resolved_seed,
        "git_commit": _get_git_commit(),
        "env_url": ENV_BASE_URL,
        "stubbed": False,
        "warnings": [],
        "overall_mean": overall_mean,
        "tasks": task_results,
    }

    # Check for suspicious scores
    warnings = detect_suspicious_scores(results)
    results["warnings"] = warnings
    if warnings:
        print("\n  ⚠ WARNINGS:", file=sys.stderr)
        for w in warnings:
            print(f"    - {w}", file=sys.stderr)

    # Write baseline_scores.json
    output_path = pathlib.Path(__file__).parent / "baseline_scores.json"
    output_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n  Wrote {output_path}", file=sys.stderr)

    # Auto-update README
    _update_readme(results)

    # Summary
    print("\n" + "=" * 60, file=sys.stderr)
    print("  BASELINE SUMMARY", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  {'Task':<14} {'Score':<8} {'Steps':<6} {'Success'}",
          file=sys.stderr)
    print(f"  {'-'*14} {'-'*8} {'-'*6} {'-'*7}", file=sys.stderr)
    for task_id, data in task_results.items():
        print(
            f"  {task_id:<14} {data['mean_score']:<8.4f} "
            f"{data['steps']:<6} {data['success']}",
            file=sys.stderr,
        )
    print(f"  {'-'*14} {'-'*8} {'-'*6} {'-'*7}", file=sys.stderr)
    print(f"  {'Overall':<14} {overall_mean:<8.4f}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()
