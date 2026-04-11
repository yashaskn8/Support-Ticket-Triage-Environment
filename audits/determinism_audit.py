#!/usr/bin/env python3
"""
determinism_audit.py
Runs multiple identical episodes to verify deterministic outputs.
"""

import os
import sys
import json
import hashlib
import httpx
import logging
from typing import Tuple, List

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

BASE_URL = os.environ.get("OPENENV_BASE_URL", "http://127.0.0.1:7860")
TASKS = ["classify", "prioritize", "resolve"]
RUNS = int(os.environ.get("DETERMINISM_RUNS", "3"))

PASS_MARK = "\033[92mPASS\033[0m"
FAIL_MARK = "\033[91mFAIL\033[0m"
ERROR_MARK = "\033[91mERROR\033[0m"

def hash_response(data: dict) -> str:
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()

def run_episode(client: httpx.Client, task_id: str) -> Tuple[List[float], List[str]]:
    try:
        reset_resp = client.post(f"{BASE_URL}/reset", json={"task_id": task_id, "seed": 42}, timeout=10.0)
        reset_resp.raise_for_status()
    except Exception as e:
        logging.error(f"[{ERROR_MARK}] Failed to reset environment for {task_id}: {e}")
        raise

    fixed_actions = {
        "classify": {"category": "TECHNICAL"},
        "prioritize": {"priority": "HIGH", "assigned_team": "tech_team", "estimated_resolution_hours": 8},
        "resolve": {
            "response_subject": "Re: Your Request",
            "response_body": "Dear Customer, I apologize. Issue will be resolved within 24 hours.",
            "internal_notes": "Standard response", "escalate": False
        }
    }
    
    rewards, hashes = [], []
    done = False
    step_count = 0
    max_steps = 10 if task_id != "resolve" else 5
    
    while not done and step_count < max_steps:
        try:
            resp = client.post(
                f"{BASE_URL}/step", 
                json={"task_id": task_id, "action": fixed_actions[task_id]},
                timeout=10.0
            )
            resp.raise_for_status()
            data = resp.json()
            
            rewards.append(round(data.get("reward", 0), 6))
            hashes.append(hash_response(data.get("observation", {})))
            done = data.get("done", False)
            step_count += 1
        except Exception as e:
            logging.error(f"[{ERROR_MARK}] Step failed during {task_id} at step {step_count}: {e}")
            raise
            
    return rewards, hashes

def main():
    all_passed = True
    
    with httpx.Client() as client:
        try:
            # Check if API is alive
            health = client.get(f"{BASE_URL}/health", timeout=5.0)
            health.raise_for_status()
        except Exception as e:
            logging.error(f"[{ERROR_MARK}] Env not reachable at {BASE_URL}. Ensure it is running: {e}")
            sys.exit(1)

        for task in TASKS:
            logging.info(f"Task: {task}")
            all_rewards, all_hashes = [], []
            
            for run in range(RUNS):
                try:
                    r, h = run_episode(client, task)
                    all_rewards.append(r)
                    all_hashes.append(h)
                    logging.info(f"  Run {run+1}: rewards={r}")
                except Exception:
                    all_passed = False
                    break # Skip remaining runs for this task if one fails
            
            if not all_rewards:
                continue
                
            if not all(r == all_rewards[0] for r in all_rewards):
                logging.info(f"  {FAIL_MARK} Rewards not identical across runs")
                all_passed = False
            if not all(h == all_hashes[0] for h in all_hashes):
                logging.info(f"  {FAIL_MARK} Observations not identical across runs")
                all_passed = False

    if all_passed:
        logging.info(f"DETERMINISM AUDIT: {PASS_MARK} FULLY DETERMINISTIC")
    else:
        logging.info(f"DETERMINISM AUDIT: {FAIL_MARK}")
        sys.exit(1)

if __name__ == "__main__":
    main()
