#!/usr/bin/env python3
"""
inference_quality_audit.py
Compares live inference against documented baselines.
"""

import json
import pathlib
import subprocess
import os
import sys
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(message)s")

TOLERANCE = 0.05
# Adjust BASELINE_PATH depending on where the script is run from.
# Assuming the script runs from the repository root or the test directory.
script_dir = pathlib.Path(__file__).parent.resolve()
root_dir = script_dir.parent
default_baseline = root_dir / "baseline_scores.json"

BASELINE_PATH = os.environ.get("BASELINE_FILE", str(default_baseline))
task_order = ["classify", "prioritize", "resolve"]

PASS_MARK = "\033[92mPASS\033[0m"
FAIL_MARK = "\033[91mFAIL\033[0m"

def parse_output(stdout: str) -> Dict[str, float]:
    scores = {}
    end_lines = [l for l in stdout.splitlines() if l.startswith("[END]")]
    for i, line in enumerate(end_lines):
        if i >= len(task_order): 
            break
        parts = dict(p.split("=", 1) for p in line[6:].split() if "=" in p)
        try:
            if "score" in parts:
                scores[task_order[i]] = float(parts.get("score", 0))
            elif "rewards" in parts and parts["rewards"]:
                rewards_list = [float(r) for r in parts["rewards"].split(",") if r.strip()]
                scores[task_order[i]] = sum(rewards_list) / len(rewards_list) if rewards_list else 0.0
            else:
                scores[task_order[i]] = 0.0
        except ValueError:
            scores[task_order[i]] = 0.0
    return scores

def main():
    baseline_file = pathlib.Path(BASELINE_PATH)
    if not baseline_file.exists():
        logging.error(f"ERROR: {BASELINE_PATH} missing")
        sys.exit(1)

    try:
        baseline = json.loads(baseline_file.read_text())
    except json.JSONDecodeError as e:
        logging.error(f"ERROR: Failed to parse {BASELINE_PATH}: {e}")
        sys.exit(1)

    logging.info("Running live inference (this may take a moment)...")
    
    inference_script = root_dir / "inference.py"
    if not inference_script.exists():
        logging.error(f"ERROR: {inference_script} not found.")
        sys.exit(1)
        
    try:
        result = subprocess.run(
            [sys.executable, str(inference_script)], 
            capture_output=True, 
            text=True, 
            env=os.environ,
            check=True
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"ERROR: inference.py failed with return code {e.returncode}")
        logging.error(e.stderr)
        sys.exit(1)

    live_scores = parse_output(result.stdout)
    all_passed = True
    
    logging.info(f"{'Task':<12} {'Baseline':>10} {'Live':>10} {'Delta':>10} {'Status':>8}")
    for t in task_order:
        try:
            baseline_score = baseline["tasks"][t]["mean_score"]
        except KeyError:
            logging.error(f"ERROR: Task '{t}' missing from baseline data. Check baseline JSON structure.")
            all_passed = False
            continue
            
        live_score = live_scores.get(t, 0.0)
        delta = live_score - baseline_score
        status = PASS_MARK if abs(delta) <= TOLERANCE else FAIL_MARK
        
        logging.info(f"{t:<12} {baseline_score:>10.3f} {live_score:>10.3f} {delta:>+10.3f} {status:>8}")
        
        if abs(delta) > TOLERANCE:
            all_passed = False
            
    if not all_passed:
        sys.exit(1)

if __name__ == "__main__":
    main()
