#!/usr/bin/env python3
"""
grader_accuracy_audit.py
Verifies that all graders produce mathematically correct outputs
for known ground-truth inputs.
"""

import sys
import logging
import traceback
from typing import Any, List, Tuple

# We import from the parent directory by modifying sys.path if needed
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server.graders.grader_classify import grade_classify
from server.graders.grader_prioritize import grade_prioritize
from server.graders.grader_resolve import grade_resolve
from server.models import ClassifyAction, PrioritizeAction, ResolveAction

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
ERROR = "\033[91mERROR\033[0m"

logging.basicConfig(level=logging.INFO, format="%(message)s")

class GraderAudit:
    def __init__(self):
        self.results: List[Tuple[str, bool]] = []
        self.errors_occurred = False

    def check(self, label: str, condition: bool, expected: Any = None, got: Any = None):
        status = PASS if condition else FAIL
        self.results.append((label, condition))
        logging.info(f"[{status}] {label}")
        if not condition:
            logging.info(f"       Expected: {expected} | Got: {got}")

    def run_classify_checks(self):
        logging.info("\n=== CLASSIFY GRADER ===")
        try:
            ticket1 = {"category": "BILLING", "subject": "invoice", "body": "billing charge"}
            res1 = grade_classify(ClassifyAction(category="BILLING"), ticket1, ticket_text="invoice billing charge")
            self.check("Exact match \u2192 1.0", res1.value == 1.0, 1.0, res1.value)

            ticket2 = {"category": "BILLING", "subject": "invoice", "body": "billing charge"}
            res2 = grade_classify(ClassifyAction(category="TECHNICAL"), ticket2, ticket_text="invoice billing charge")
            self.check("Cross-group mismatch \u2192 0.015", abs(res2.value - 0.015) < 0.001, 0.015, res2.value)
        except Exception as e:
            logging.error(f"[{ERROR}] Failed to execute classify checks: {e}")
            logging.error(traceback.format_exc())
            self.errors_occurred = True

    def run_prioritize_checks(self):
        logging.info("\n=== PRIORITIZE GRADER ===")
        try:
            # We construct a ticket that guarantees HIGH and tech_team
            ticket3 = {"subject": "api error", "body": "please fix", "previous_interactions": 0}
            r = grade_prioritize(
                PrioritizeAction(priority="HIGH", assigned_team="tech_team", estimated_resolution_hours=8),
                ticket3
            )
            self.check("Perfect prioritize \u2192 1.0", abs(r.value - 1.0) < 0.001, 1.0, r.value)
        except Exception as e:
            logging.error(f"[{ERROR}] Failed to execute prioritize checks: {e}")
            logging.error(traceback.format_exc())
            self.errors_occurred = True

    def run_resolve_checks(self):
        logging.info("\n=== RESOLVE GRADER ===")
        body = ("Dear Customer, I sincerely apologize for the disruption. "
                "Your refund will be processed within 5-7 business days. "
                "Our team has escalated this to billing.")
        try:
            sample_ticket = {"subject": "Refund", "body": "Refund"}
            r = grade_resolve(
                ResolveAction(
                    response_subject="Re: Refund",
                    response_body=body,
                    internal_notes="Escalated",
                    escalate=False
                ),
                ticket=sample_ticket,
                ticket_subject="Refund",
                ticket_priority="MEDIUM",
                ticket_previous_interactions=1,
                kb_articles=[],
                ground_truth={"required_elements": ["refund", "apologize", "team"],
                 "forbidden_elements": ["lawsuit"],
                 "priority": "MEDIUM",
                 "previous_interactions": 1,
                 "knowledge_base": [],
                 "ticket": sample_ticket}
            )
            structure_score = r.breakdown.get("structure", 0)
            self.check("Resolve structure \u2192 1.0", abs(structure_score - 1.0) < 0.001, 1.0, structure_score)
        except Exception as e:
            logging.error(f"[{ERROR}] Failed to execute resolve checks: {e}")
            logging.error(traceback.format_exc())
            self.errors_occurred = True

    def run_all(self):
        self.run_classify_checks()
        self.run_prioritize_checks()
        self.run_resolve_checks()
        
        logging.info("\nSUMMARY")
        passed = sum(1 for _, v in self.results if v)
        total = len(self.results)
        
        if total == 0:
            logging.error("No checks were run (probably due to errors).")
            return False
            
        logging.info(f"Grader Accuracy: {passed}/{total} checks passed")
        return passed == total and not self.errors_occurred

def main():
    audit = GraderAudit()
    success = audit.run_all()
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
