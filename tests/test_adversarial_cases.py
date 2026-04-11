import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.graders.grader_classify import _compute_target_category
from server.graders.grader_prioritize import _compute_true_priority

def test_adversarial_billing_downgrade():
    """
    TICKET 1 (Billing — Adversarial)
    Tests that a downgrade inquiry doesn't get artificially inflated to HIGH/CRITICAL 
    just because it mentions 'downgrade' or 'charge'. It has no extreme urgency signals.
    """
    ticket = {
        "subject": "Unexpected additional charge of $79.00 after downgrade \u2014 ref INV-88321",
        "body": "We downgraded our plan on May 2nd, but I still see a charge of $79.00 on May 3rd tied to invoice INV-88321. Based on your pricing page, the downgrade should have reduced our cost immediately. This is confusing because our usage also dropped significantly after the downgrade. Our account ID is AC-44192. Please clarify why this charge was applied and whether it will be adjusted.",
        "customer_name": "Diego Alvarez"
    }
    
    category = _compute_target_category(ticket)
    priority = _compute_true_priority(ticket, category)
    
    assert category == "BILLING", f"Expected BILLING, got {category}"
    assert priority == "MEDIUM", f"Expected MEDIUM, got {priority}"

def test_adversarial_billing_high_pressure():
    """
    TICKET 2 (Billing — High Pressure)
    Tests that an urgent audit/duplicate payment issue correctly spikes to CRITICAL
    due to compounded financial risk and strict deadline phrasing.
    """
    ticket = {
        "subject": "Duplicate payment of $1,200 \u2014 urgent before audit cutoff",
        "body": "Two payments of $1,200 were processed for the same invoice INV-99211 within 6 hours on June 10th. Our finance audit closes tomorrow at 14:00, and this discrepancy must be resolved before then.",
        "customer_name": "Fatima Zahra"
    }
    
    category = _compute_target_category(ticket)
    priority = _compute_true_priority(ticket, category)
    
    assert category == "BILLING", f"Expected BILLING, got {category}"
    assert priority == "CRITICAL", f"Expected CRITICAL, got {priority}"

def test_adversarial_technical_silent_critical():
    """
    TICKET 3 (Technical — Silent Critical)
    Tests a ticket that describes a catastrophic failure but with calm/informal phrasing,
    checking if the system correctly identifies it as CRITICAL or HIGH regardless of tone.
    """
    ticket = {
        "subject": "Main production API endpoint returning 503 since deploy",
        "body": "Hey team, just wanted to let you know that our main production API is down and every request is returning a 503 error. Nothing is working right now. Let us know when it's back.",
        "customer_name": "Sam"
    }
    
    category = _compute_target_category(ticket)
    priority = _compute_true_priority(ticket, category)
    
    assert category == "TECHNICAL", f"Expected TECHNICAL, got {category}"
    assert priority in ["HIGH", "CRITICAL"], f"Expected HIGH or CRITICAL, got {priority}"
