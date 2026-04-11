"""
Submission readiness check for the Support Triage OpenEnv environment.

Performs a definitive go/no-go validation of all submission requirements.
Run this as the final step before git push to verify every requirement
is met. Exits with code 0 on pass, 1 on any failure.

Usage:
    python audits/submission_readiness.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import importlib

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def _check(name: str, passed: bool, detail: str = "") -> bool:
    """
    Print a check result and return whether it passed.

    Args:
        name: Name of the check.
        passed: Whether the check passed.
        detail: Optional detail message.

    Returns:
        True if passed, False otherwise.
    """
    icon = "[PASS]" if passed else "[FAIL]"
    suffix = f" — {detail}" if detail else ""
    print(f"  {icon} {name}{suffix}")
    return passed


def run_readiness_checks() -> bool:
    """
    Run all submission readiness checks.

    Checks:
      1. openenv.yaml exists and is valid YAML
      2. Dockerfile exists
      3. requirements.txt exists
      4. All three task modules import cleanly
      5. All three grader modules import cleanly
      6. models.py EnvironmentState has analytics fields
      7. baseline_scores.json exists and is not stubbed
      8. README.md contains required sections
      9. /data-source endpoint is registered in app.py
     10. Resolve grader WEIGHTS sum to 1.0
     11. Classify grader has incentive ordering assertion
     12. environment.py has trajectory bonus method

    Returns:
        True if all checks pass, False otherwise.
    """
    root = pathlib.Path(__file__).parent.parent
    results: list[bool] = []

    print("\n" + "=" * 60)
    print("  SUBMISSION READINESS CHECK")
    print("=" * 60 + "\n")

    # ── 1. openenv.yaml ──
    print("[1/12] OpenEnv Specification")
    yaml_path = root / "openenv.yaml"
    yaml_exists = yaml_path.exists()
    results.append(_check("openenv.yaml exists", yaml_exists))
    if yaml_exists:
        try:
            import yaml
            with yaml_path.open() as f:
                spec = yaml.safe_load(f)
            results.append(_check("openenv.yaml is valid YAML", True))
            tasks = spec.get("tasks", [])
            results.append(_check(
                f"Contains {len(tasks)} tasks",
                len(tasks) == 3,
                f"expected 3, got {len(tasks)}",
            ))
            api = spec.get("api", {})
            has_ds = "data_source_endpoint" in api
            results.append(_check("/data-source in API spec", has_ds))
            has_rm = "reward_mechanics" in spec
            results.append(_check("reward_mechanics block present", has_rm))
        except Exception as e:
            results.append(_check("openenv.yaml parsing", False, str(e)))

    # ── 2. Dockerfile ──
    print("\n[2/12] Container Configuration")
    results.append(_check(
        "Dockerfile exists",
        (root / "Dockerfile").exists(),
    ))

    # ── 3. requirements.txt ──
    print("\n[3/12] Dependencies")
    results.append(_check(
        "requirements.txt exists",
        (root / "requirements.txt").exists(),
    ))

    # ── 4. Task modules ──
    print("\n[4/12] Task Modules")
    for task_name in ["classify", "prioritize", "resolve"]:
        try:
            mod = importlib.import_module(f"server.tasks.task_{task_name}")
            results.append(_check(f"task_{task_name}.py imports cleanly", True))
        except Exception as e:
            results.append(_check(f"task_{task_name}.py imports cleanly", False, str(e)))

    # ── 5. Grader modules ──
    print("\n[5/12] Grader Modules")
    for grader_name in ["classify", "prioritize", "resolve"]:
        try:
            mod = importlib.import_module(f"server.graders.grader_{grader_name}")
            results.append(_check(f"grader_{grader_name}.py imports cleanly", True))
        except Exception as e:
            results.append(_check(f"grader_{grader_name}.py imports cleanly", False, str(e)))

    # ── 6. EnvironmentState analytics fields ──
    print("\n[6/12] Environment State Analytics")
    try:
        from server.models import EnvironmentState
        es = EnvironmentState(
            task_id="test",
            current_ticket_index=0,
            total_tickets=1,
            episode_id="test-id",
        )
        analytics_fields = [
            "mean_reward_so_far", "min_reward_this_episode",
            "max_reward_this_episode", "penalties_applied_total",
            "steps_remaining",
        ]
        for field in analytics_fields:
            results.append(_check(
                f"EnvironmentState.{field} exists",
                hasattr(es, field),
            ))
    except Exception as e:
        results.append(_check("EnvironmentState import", False, str(e)))

    # ── 7. Baseline scores ──
    print("\n[7/12] Baseline Scores")
    baseline_path = root / "baseline_scores.json"
    if baseline_path.exists():
        try:
            data = json.loads(baseline_path.read_text(encoding="utf-8"))
            is_stubbed = data.get("stubbed", True)
            results.append(_check(
                "baseline_scores.json not stubbed",
                not is_stubbed,
                "stubbed=True (run baseline_runner.py first)" if is_stubbed else "verified",
            ))
        except Exception as e:
            results.append(_check("baseline_scores.json valid JSON", False, str(e)))
    else:
        results.append(_check(
            "baseline_scores.json exists",
            False,
            "run baseline_runner.py first",
        ))

    # ── 8. README sections ──
    print("\n[8/12] README Documentation")
    readme_path = root / "README.md"
    if readme_path.exists():
        readme_text = readme_path.read_text(encoding="utf-8")
        required_sections = [
            "What Makes This Environment Different",
            "Development Setup",
            "Trajectory Consistency Bonus",
            "/data-source",
        ]
        for section in required_sections:
            results.append(_check(
                f"README contains '{section}'",
                section in readme_text,
            ))
    else:
        results.append(_check("README.md exists", False))

    # ── 9. /data-source endpoint ──
    print("\n[9/12] Data Source Endpoint")
    try:
        app_path = root / "server" / "app.py"
        app_text = app_path.read_text(encoding="utf-8")
        results.append(_check(
            "/data-source route in app.py",
            "/data-source" in app_text,
        ))
        results.append(_check(
            "source_metadata() in fetcher.py",
            "source_metadata" in (root / "server" / "data" / "fetcher.py").read_text(encoding="utf-8"),
        ))
    except Exception as e:
        results.append(_check("Endpoint check", False, str(e)))

    # ── 10. Resolve grader weights ──
    print("\n[10/12] Resolve Grader Weights")
    try:
        from server.graders.grader_resolve import WEIGHTS
        total = sum(WEIGHTS.values())
        results.append(_check(
            f"WEIGHTS sum = {total:.4f}",
            abs(total - 1.0) < 1e-9,
        ))
        results.append(_check(
            f"Contains {len(WEIGHTS)} dimensions",
            len(WEIGHTS) == 9,
            f"expected 9, got {len(WEIGHTS)}",
        ))
    except Exception as e:
        results.append(_check("WEIGHTS check", False, str(e)))

    # ── 11. Classify grader assertion ──
    print("\n[11/12] Classify Grader Incentive Ordering")
    try:
        classify_text = (root / "server" / "graders" / "grader_classify.py").read_text(encoding="utf-8")
        has_assertion = "assert 1.0 > 0.65 > 0.15" in classify_text
        results.append(_check(
            "Incentive ordering assertion present",
            has_assertion,
        ))
    except Exception as e:
        results.append(_check("Assertion check", False, str(e)))

    # ── 12. Trajectory bonus ──
    print("\n[12/12] Trajectory Consistency Bonus")
    try:
        env_text = (root / "server" / "environment.py").read_text(encoding="utf-8")
        has_method = "_compute_trajectory_bonus" in env_text
        results.append(_check(
            "_compute_trajectory_bonus() in environment.py",
            has_method,
        ))
        has_criteria = all(c in env_text for c in [
            "monotonic improvement",
            "catastrophic",
            "variance",
            "above-baseline",
        ])
        results.append(_check(
            "All four trajectory criteria documented",
            has_criteria,
        ))
    except Exception as e:
        results.append(_check("Trajectory bonus check", False, str(e)))

    # ── Summary ──
    total = len(results)
    passed = sum(results)
    failed = total - passed

    print("\n" + "=" * 60)
    if failed == 0:
        print(f"  ALL {total} CHECKS PASSED — READY TO SUBMIT")
    else:
        print(f"  {passed}/{total} PASSED, {failed} FAILED")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_readiness_checks()
    sys.exit(0 if success else 1)
