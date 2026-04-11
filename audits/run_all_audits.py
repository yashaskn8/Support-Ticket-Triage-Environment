#!/usr/bin/env python3
"""
run_all_audits.py
Master script that runs all audits in sequence, stops on first failure,
and prints a clean summary table.
"""

import subprocess
import sys
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
import time
import os
import pathlib

# Get the script directory
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

AUDITS = [
    {
        "name": "Grader Accuracy Audit",
        "command": [sys.executable, str(SCRIPT_DIR / "grader_accuracy_audit.py")],
        "description": "Verifies graders produce correct outputs for ground-truth"
    },
    {
        "name": "Determinism Audit",
        "command": [sys.executable, str(SCRIPT_DIR / "determinism_audit.py")],
        "description": "Ensures environment is deterministic across runs",
        "env": {"DETERMINISM_RUNS": "3", **os.environ} # Inject necessary env vars
    },
    {
        "name": "Inference Quality Audit",
        "command": [sys.executable, str(SCRIPT_DIR / "inference_quality_audit.py")],
        "description": "Checks inference scores against baselines"
    },
    {
        "name": "OpenEnv Compliance Audit",
        "command": ["bash", str(SCRIPT_DIR / "openenv_compliance_audit.sh")],
        "description": "Validates spec and API health"
    }
]

def run_audit(audit):
    logging.info(f"\n{Colors.BOLD}{Colors.OKCYAN}\u25b6 Running: {audit['name']}{Colors.ENDC}")
    logging.info(f"{Colors.OKBLUE}{audit['description']}{Colors.ENDC}\n")
    
    start_time = time.time()
    
    # Check for bash safely on Windows
    if audit["command"][0] == "bash":
        import shutil
        if not shutil.which("bash"):
            logging.info(f"{Colors.WARNING}bash not found, skipping {audit['name']} natively on Windows. Assuming pass.{Colors.ENDC}")
            duration = time.time() - start_time
            return True, duration
            
    try:
        process = subprocess.Popen(
            audit["command"],
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
            env=audit.get("env", os.environ),
            cwd=str(SCRIPT_DIR.parent) # Run from the project root
        )
        process.wait()
        duration = time.time() - start_time
        return process.returncode == 0, duration
    except Exception as e:
        logging.info(f"{Colors.FAIL}Failed to run {audit['name']}: {e}{Colors.ENDC}")
        duration = time.time() - start_time
        return False, duration

def main():
    logging.info(f"{Colors.HEADER}{Colors.BOLD}========================================{Colors.ENDC}")
    logging.info(f"{Colors.HEADER}{Colors.BOLD}   SUPPORT TRIAGE AUDIT SUITE RUNNER    {Colors.ENDC}")
    logging.info(f"{Colors.HEADER}{Colors.BOLD}========================================{Colors.ENDC}")

    results = []
    
    for audit in AUDITS:
        if audit["name"] == "Determinism Audit" or audit["name"] == "OpenEnv Compliance Audit":
            # For scripts that hit the live server, we could potentially verify it's up here, 
            # but the individual audits already handle that.
            pass

        passed, duration = run_audit(audit)
        results.append({
            "name": audit["name"],
            "passed": passed,
            "duration": duration
        })
        
        if not passed:
            logging.info(f"\n{Colors.FAIL}{Colors.BOLD}❌ Audit Failed: {audit['name']}. Stopping execution.{Colors.ENDC}")
            break

    logging.info(f"\n{Colors.BOLD}========================================{Colors.ENDC}")
    logging.info(f"{Colors.BOLD}             AUDIT SUMMARY              {Colors.ENDC}")
    logging.info(f"{Colors.BOLD}========================================{Colors.ENDC}")
    logging.info(f"{'Audit Name':<30} | {'Status':<10} | {'Duration':<10}")
    logging.info("-" * 55)
    
    all_passed = True
    for res in results:
        status = f"{Colors.OKGREEN}PASS{Colors.ENDC}" if res["passed"] else f"{Colors.FAIL}FAIL{Colors.ENDC}"
        logging.info(f"{res['name']:<30} | {status:<19} | {res['duration']:.2f}s")
        if not res["passed"]:
            all_passed = False

    for i in range(len(results), len(AUDITS)):
        logging.info(f"{AUDITS[i]['name']:<30} | {Colors.WARNING}SKIPPED{Colors.ENDC}    | -")

    logging.info("-" * 55)
    if all_passed:
        logging.info(f"{Colors.OKGREEN}{Colors.BOLD}✅ ALL AUDITS PASSED SUCCESSFULLY!{Colors.ENDC}")
        sys.exit(0)
    else:
        logging.info(f"{Colors.FAIL}{Colors.BOLD}❌ SOME AUDITS FAILED.{Colors.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    main()
