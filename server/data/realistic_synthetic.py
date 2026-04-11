"""
Realistic synthetic support ticket source for the Support Triage Environment.

Provides 30 curated, realistic synthetic support tickets — 6 per category —
that are indistinguishable from live enterprise support data in vocabulary,
tone, and structure. Used as the secondary source when GitHub is rate-limited,
replacing the JSONPlaceholder source entirely.

Design properties:
  - Every ticket is 4–8 sentences long.
  - Every ticket includes a specific detail: dollar amount, order number,
    timestamp, error code, or account identifier.
  - Zero occurrences of category label strings (BILLING, TECHNICAL, ACCOUNT,
    SHIPPING, GENERAL) in subject or body.
  - Customer names reflect global diversity.
  - Email domains are fictional but plausible.
"""

from __future__ import annotations

import os
import random
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.models import Ticket


class RealisticSyntheticSource:
    """
    A curated pool of 30 realistic synthetic support tickets used when
    GitHub Issues API is unavailable due to rate limiting.

    Each ticket is designed to be indistinguishable from live enterprise
    support data with specific details, globally diverse customer names,
    and realistic language patterns.

    Ground truth labels are embedded in each ticket dict under the
    ground_truth_ prefix and used to construct LabeledTicket objects.
    """

    TICKETS: List[dict] = [
        # ── BILLING CATEGORY (6 tickets) ─────────────────────────────────
        {
            "subject": "Duplicate charge of $149.00 on March invoice \u2014 ref INV-2024-0341",
            "body": (
                "I have just reviewed my credit card statement for March and "
                "noticed two identical charges of $149.00 on the 3rd and 4th "
                "of the month, both referencing invoice number INV-2024-0341. "
                "My invoicing cycle has always been on the 1st, so the second "
                "charge on the 4th appears to be an error. I have not made "
                "any plan changes or upgrades this month. My user profile email "
                "is registered under this address. Could you please investigate "
                "and confirm whether a reversal will be issued, and if so, "
                "within what timeframe I should expect to see it on my statement?"
            ),
            "customer_name": "Priya Venkataraman",
            "customer_email": "p.venkataraman@nexaflow-corp.com",
            "previous_interactions": 0,
            "ground_truth_category": "BILLING",
            "ground_truth_priority": "HIGH",
            "ground_truth_team": "billing_team",
            "ground_truth_resolution_hours": 8,
        },
        {
            "subject": "Annual plan renewal charged at wrong rate \u2014 expected $599, billed $899",
            "body": (
                "Our annual renewal was processed yesterday at $899, but the "
                "rate I agreed to when we upgraded in November was $599 per "
                "year, locked in for 24 months. I have the original confirmation "
                "email with the agreed pricing. This is a discrepancy of $300 "
                "and I would like either a corrected invoice or a partial refund "
                "of $300 to the Mastercard ending in 7823 on file. The renewal "
                "confirmation number is REN-98412. Please advise on your process "
                "for resolving invoicing discrepancies and the expected resolution "
                "timeline."
            ),
            "customer_name": "Olumide Fashola",
            "customer_email": "o.fashola@brightledger.io",
            "previous_interactions": 1,
            "ground_truth_category": "BILLING",
            "ground_truth_priority": "HIGH",
            "ground_truth_team": "billing_team",
            "ground_truth_resolution_hours": 8,
        },
        {
            "subject": "Cannot download VAT invoice for Q1 \u2014 urgent for audit submission",
            "body": (
                "We are undergoing an external audit and our auditor requires "
                "all VAT-compliant invoices for Q1 2024 by the end of this week. "
                "When I navigate to the invoicing portal and select the date range "
                "January through March, the export button produces a blank PDF "
                "with no line items. I have tried three different browsers and "
                "two different user profiles with invoicing admin permissions. "
                "This is blocking our audit submission and carries a compliance "
                "deadline. Our user profile number is ACC-55021."
            ),
            "customer_name": "Astrid Lindqvist",
            "customer_email": "a.lindqvist@stackvault.net",
            "previous_interactions": 2,
            "ground_truth_category": "BILLING",
            "ground_truth_priority": "CRITICAL",
            "ground_truth_team": "billing_team",
            "ground_truth_resolution_hours": 2,
        },
        {
            "subject": "Refund not received after 14 days \u2014 case number REF-20241",
            "body": (
                "I submitted a refund request on the 8th of last month for "
                "a charge of $79.00 and was told the funds would appear within "
                "7\u201310 business days. It has now been 14 business days and "
                "nothing has appeared on my statement. I checked with my bank "
                "and they have no record of a pending reversal. My case number "
                "is REF-20241. At this point I need either confirmation that "
                "the refund was processed with a bank reference number, or "
                "an escalation to someone who can resolve this today."
            ),
            "customer_name": "Marcus Tetteh",
            "customer_email": "marcus.tetteh@personalmail-example.gh",
            "previous_interactions": 3,
            "ground_truth_category": "BILLING",
            "ground_truth_priority": "HIGH",
            "ground_truth_team": "billing_team",
            "ground_truth_resolution_hours": 4,
        },
        {
            "subject": "Request to update payment method before next invoicing cycle",
            "body": (
                "The credit card currently on file for our user profile expires at "
                "the end of this month and I would like to update it before "
                "the next invoicing cycle on the 15th to avoid any interruption "
                "to our service. When I navigate to the payment methods section "
                "of the user profile portal, the page loads but the 'Add new card' "
                "button does not respond to clicks. I have tested this in "
                "Chrome and Safari on two different machines. Our user profile ID "
                "is AC-774209."
            ),
            "customer_name": "Yuki Hashimoto",
            "customer_email": "y.hashimoto@cloudpulse-example.jp",
            "previous_interactions": 0,
            "ground_truth_category": "BILLING",
            "ground_truth_priority": "MEDIUM",
            "ground_truth_team": "billing_team",
            "ground_truth_resolution_hours": 24,
        },
        {
            "subject": "Need clarification on overage charges for April \u2014 $220 above expected",
            "body": (
                "Our April invoice came in at $220 higher than our contracted "
                "monthly rate and I cannot find a clear breakdown of what "
                "generated the overage. The invoice line items show a single "
                "entry labelled 'Usage overage \u2014 April' with no further "
                "detail. Our contract specifies a monthly limit of 500,000 "
                "API calls and I would like to see the actual usage report "
                "that justifies this charge before authorising payment. "
                "Invoice number is INV-2024-0489."
            ),
            "customer_name": "Carolina Restrepo",
            "customer_email": "c.restrepo@datastream-example.co",
            "previous_interactions": 1,
            "ground_truth_category": "BILLING",
            "ground_truth_priority": "MEDIUM",
            "ground_truth_team": "billing_team",
            "ground_truth_resolution_hours": 24,
        },
        # ── TECHNICAL CATEGORY (6 tickets) ───────────────────────────────
        {
            "subject": "Webhook delivery failing with HTTP 422 since yesterday 16:00 UTC",
            "body": (
                "Since approximately 16:00 UTC yesterday our webhook endpoint "
                "has been receiving malformed payloads that trigger HTTP 422 "
                "validation errors on our end. The payload structure appears "
                "to have changed \u2014 specifically the 'event_type' field is now "
                "nested under a 'metadata' object rather than at the top level, "
                "which breaks our existing validation schema. This is affecting "
                "100 percent of webhook deliveries and blocking our order "
                "processing pipeline. Was there an undocumented API schema "
                "change deployed yesterday afternoon? Our integration ID is "
                "INT-6612."
            ),
            "customer_name": "Dmitri Volkov",
            "customer_email": "d.volkov@infranode-example.ru",
            "previous_interactions": 0,
            "ground_truth_category": "TECHNICAL",
            "ground_truth_priority": "CRITICAL",
            "ground_truth_team": "tech_team",
            "ground_truth_resolution_hours": 2,
        },
        {
            "subject": "SDK throwing NullPointerException on initialisation \u2014 version 3.4.1",
            "body": (
                "After upgrading from SDK version 3.3.8 to 3.4.1 this morning, "
                "our application throws a NullPointerException on line 84 of "
                "AuthManager.java during the initialisation sequence. The full "
                "stack trace is: java.lang.NullPointerException at "
                "com.yoursdk.auth.AuthManager.init(AuthManager.java:84). "
                "Downgrading back to 3.3.8 resolves the issue immediately, "
                "confirming this is a regression introduced in 3.4.1. We "
                "cannot stay on the older version indefinitely as it lacks "
                "a security patch we need. Is there a hotfix available or "
                "a known workaround?"
            ),
            "customer_name": "Wei Chen",
            "customer_email": "wei.chen@bytecraft-example.cn",
            "previous_interactions": 1,
            "ground_truth_category": "TECHNICAL",
            "ground_truth_priority": "HIGH",
            "ground_truth_team": "tech_team",
            "ground_truth_resolution_hours": 8,
        },
        {
            "subject": "Dashboard reports showing stale data \u2014 last refresh 31 hours ago",
            "body": (
                "The analytics dashboard for our user profile (ACC-88201) has not "
                "refreshed since 09:14 UTC yesterday, now over 31 hours ago. "
                "All metrics are frozen at that timestamp despite our data "
                "pipeline confirming that events are being successfully "
                "ingested via the API \u2014 we can see them in our own logs. "
                "The dashboard status indicator shows 'Live' but the data "
                "is clearly not updating. This is impacting our ability to "
                "monitor production metrics in real time. Is there a known "
                "processing delay or a stuck job on your end?"
            ),
            "customer_name": "Amara Diallo",
            "customer_email": "a.diallo@metricshub-example.sn",
            "previous_interactions": 2,
            "ground_truth_category": "TECHNICAL",
            "ground_truth_priority": "HIGH",
            "ground_truth_team": "tech_team",
            "ground_truth_resolution_hours": 6,
        },
        {
            "subject": "OAuth token exchange returning 'invalid_grant' intermittently",
            "body": (
                "Approximately 15 percent of OAuth token exchange requests "
                "from our mobile application are returning an 'invalid_grant' "
                "error despite the authorisation code being fresh and unused. "
                "The issue is non-deterministic \u2014 the same code sometimes "
                "succeeds and sometimes fails within the 60-second validity "
                "window. Our OAuth app ID is APP-33091. We have ruled out "
                "clock skew on our side as the issue persists even when the "
                "exchange is attempted within 5 seconds of code issuance. "
                "This is causing intermittent login failures for approximately "
                "1 in 7 users."
            ),
            "customer_name": "Isabela Ferreira",
            "customer_email": "i.ferreira@mobilestack-example.br",
            "previous_interactions": 0,
            "ground_truth_category": "TECHNICAL",
            "ground_truth_priority": "HIGH",
            "ground_truth_team": "tech_team",
            "ground_truth_resolution_hours": 8,
        },
        {
            "subject": "CSV export truncating rows at exactly 10,000 \u2014 data loss concern",
            "body": (
                "Every CSV export from the reporting module is truncating at "
                "exactly 10,000 rows regardless of the actual dataset size. "
                "Our March dataset contains 47,832 records but the exported "
                "file contains exactly 10,000 rows with no error message or "
                "truncation warning. This appears to be a hardcoded limit "
                "that was not present in the previous version. We discovered "
                "this issue during a data reconciliation and are concerned "
                "about data integrity for exports we ran over the past "
                "three weeks. Is there a way to export the full dataset "
                "and were previous exports also affected?"
            ),
            "customer_name": "Tomasz Kowalczyk",
            "customer_email": "t.kowalczyk@datapipe-example.pl",
            "previous_interactions": 1,
            "ground_truth_category": "TECHNICAL",
            "ground_truth_priority": "MEDIUM",
            "ground_truth_team": "tech_team",
            "ground_truth_resolution_hours": 12,
        },
        {
            "subject": "Question about rate limiting behaviour on the search endpoint",
            "body": (
                "We are designing our integration around the search endpoint "
                "and I have a question about the rate limiting behaviour "
                "before we go to production. The documentation states "
                "'100 requests per minute' but does not clarify whether "
                "this is a sliding window or a fixed 60-second reset. "
                "We need to know this to design our retry backoff correctly. "
                "Additionally, when the rate limit is exceeded, the response "
                "body contains only an HTTP 429 status with no Retry-After "
                "header \u2014 is this intentional or will the header be added "
                "in a future release? Our integration is targeting production "
                "in two weeks."
            ),
            "customer_name": "Nguyen Thi Lan",
            "customer_email": "lan.nguyen@devharbor-example.vn",
            "previous_interactions": 0,
            "ground_truth_category": "TECHNICAL",
            "ground_truth_priority": "LOW",
            "ground_truth_team": "tech_team",
            "ground_truth_resolution_hours": 48,
        },
        # ── ACCOUNT CATEGORY (6 tickets) ─────────────────────────────────
        {
            "subject": "Locked after failed 2FA attempts \u2014 production access lost",
            "body": (
                "My user profile was locked at 08:43 this morning after three "
                "failed two-factor authentication attempts. I had recently "
                "changed phones and the authenticator app was still bound "
                "to my old device which I no longer have access to. I have "
                "tried the backup codes provided during initial 2FA setup "
                "but the system reports them as already used. This user profile "
                "manages our production infrastructure and the lockout is "
                "preventing deployment operations. My user profile ID is "
                "USR-441029 and the registered email is this address. "
                "I need identity-verified user profile recovery urgently."
            ),
            "customer_name": "Sione Tuilagi",
            "customer_email": "s.tuilagi@cloudops-example.nz",
            "previous_interactions": 0,
            "ground_truth_category": "ACCOUNT",
            "ground_truth_priority": "CRITICAL",
            "ground_truth_team": "account_team",
            "ground_truth_resolution_hours": 2,
        },
        {
            "subject": "Suspicious login from Singapore IP \u2014 did not authorise this access",
            "body": (
                "I received a login alert at 03:22 this morning for an "
                "access event from IP address 103.28.54.211, which "
                "geolocates to Singapore. I am based in Germany and have "
                "never accessed this user profile from that region. I immediately "
                "changed my password but I need to know whether any data "
                "was accessed or any settings were modified during that "
                "session. My user profile manages API keys for three production "
                "systems, so if those keys were viewed I need to rotate "
                "them immediately. Please provide a session activity log "
                "for the last 24 hours."
            ),
            "customer_name": "Franziska Bauer",
            "customer_email": "f.bauer@securenet-example.de",
            "previous_interactions": 1,
            "ground_truth_category": "ACCOUNT",
            "ground_truth_priority": "HIGH",
            "ground_truth_team": "account_team",
            "ground_truth_resolution_hours": 4,
        },
        {
            "subject": "SSO configuration broken after domain migration \u2014 200 users affected",
            "body": (
                "We completed a domain migration from oldcompany.com to "
                "newcompany.io last Friday and since then our SAML SSO "
                "integration has been broken. The entity ID and ACS URL "
                "in our identity provider still reference the old domain. "
                "I have updated them in our IdP configuration but the "
                "service still returns 'SAML authentication failed' "
                "with error code SAML-4021. Approximately 200 users are "
                "unable to log in via SSO and are currently blocked from "
                "accessing the platform. Is there a configuration update "
                "needed on your side to reflect the new domain?"
            ),
            "customer_name": "Rohan Mehta",
            "customer_email": "r.mehta@newcompany.io",
            "previous_interactions": 3,
            "ground_truth_category": "ACCOUNT",
            "ground_truth_priority": "CRITICAL",
            "ground_truth_team": "account_team",
            "ground_truth_resolution_hours": 2,
        },
        {
            "subject": "Need to transfer user profile ownership before employee departure",
            "body": (
                "One of our senior engineers, who is the current user profile "
                "owner for workspace WS-20931, is leaving the company on "
                "Friday. I need to transfer user profile ownership to myself "
                "before then to avoid losing administrative access to our "
                "workspace. I am already a member of the workspace with "
                "admin permissions, but the ownership transfer option in "
                "the settings panel is greyed out and shows the tooltip "
                "'Contact support to transfer ownership'. What is the "
                "process and how quickly can this be completed? The "
                "departing employee has confirmed they can verify their "
                "identity if required."
            ),
            "customer_name": "Kenji Watanabe",
            "customer_email": "k.watanabe@techventures-example.jp",
            "previous_interactions": 0,
            "ground_truth_category": "ACCOUNT",
            "ground_truth_priority": "HIGH",
            "ground_truth_team": "account_team",
            "ground_truth_resolution_hours": 6,
        },
        {
            "subject": "Password reset emails not arriving \u2014 checked spam, issue persists for 4 days",
            "body": (
                "I have been trying to reset my password for four days and "
                "none of the reset emails are arriving. I have checked my "
                "spam and junk folders thoroughly, added your sending "
                "domain to my safe senders list, and tried three separate "
                "password reset requests. Our IT team has confirmed there "
                "is no email filtering rule blocking your domain at the "
                "corporate level. Is it possible the user profile is associated "
                "with a slightly different email address, or is there a "
                "delivery issue on your end? My user ID is USR-882341."
            ),
            "customer_name": "Adaeze Okonkwo",
            "customer_email": "a.okonkwo@fintech-example.ng",
            "previous_interactions": 2,
            "ground_truth_category": "ACCOUNT",
            "ground_truth_priority": "MEDIUM",
            "ground_truth_team": "account_team",
            "ground_truth_resolution_hours": 8,
        },
        {
            "subject": "Request to update invoicing email address on user profile",
            "body": (
                "I would like to update the invoicing contact email address "
                "on our user profile from the current address (finance-old@company.com) "
                "to our new finance team address (finance@newcompany.io). "
                "I am an user profile admin and have tried updating this in "
                "the profile settings but the invoicing email field appears "
                "to be read-only for admin users \u2014 only the user profile owner "
                "can modify it. Our user profile owner is currently travelling "
                "and unavailable for two weeks, but we have an invoice "
                "due in five days that needs to go to the new address. "
                "Is there an alternative process for this update?"
            ),
            "customer_name": "Leila Hosseini",
            "customer_email": "l.hosseini@newcompany.io",
            "previous_interactions": 0,
            "ground_truth_category": "ACCOUNT",
            "ground_truth_priority": "MEDIUM",
            "ground_truth_team": "account_team",
            "ground_truth_resolution_hours": 24,
        },
        # ── SHIPPING CATEGORY (6 tickets) ────────────────────────────────
        {
            "subject": "Order #ORD-774821 delivered to wrong address \u2014 tracking shows signed by unknown person",
            "body": (
                "My order ORD-774821 was marked as delivered yesterday at "
                "14:33 with a signature from 'J. Morrison'. I live alone "
                "and was home all day \u2014 no delivery was made to my address. "
                "The tracking map shows the delivery point is approximately "
                "0.3 miles from my registered address. I believe it was "
                "delivered to the wrong location. The order contained "
                "equipment worth $340 and I need either the item recovered "
                "and redelivered correctly, or a full replacement dispatched "
                "immediately. I cannot wait more than two days as this is "
                "for a client project."
            ),
            "customer_name": "Nadia Osei",
            "customer_email": "n.osei@designlab-example.gh",
            "previous_interactions": 1,
            "ground_truth_category": "SHIPPING",
            "ground_truth_priority": "HIGH",
            "ground_truth_team": "logistics_team",
            "ground_truth_resolution_hours": 8,
        },
        {
            "subject": "Fragile item arrived broken despite 'fragile' label \u2014 order #ORD-330921",
            "body": (
                "Order ORD-330921 arrived this morning with significant "
                "damage to the contents. The outer box was intact but the "
                "inner packaging was clearly insufficient for a fragile "
                "item \u2014 there was no foam padding or void fill despite "
                "the product page stating 'ships in protective packaging'. "
                "I have photographs of the damaged item and the packaging. "
                "The item is a calibrated measurement instrument worth "
                "$890 that is now unusable. I need a replacement sent "
                "with appropriate packaging, and I would like to understand "
                "how this will be prevented on the replacement shipment."
            ),
            "customer_name": "Bjorn Eriksson",
            "customer_email": "b.eriksson@labtools-example.se",
            "previous_interactions": 0,
            "ground_truth_category": "SHIPPING",
            "ground_truth_priority": "HIGH",
            "ground_truth_team": "logistics_team",
            "ground_truth_resolution_hours": 6,
        },
        {
            "subject": "Order placed 18 days ago still showing 'processing' \u2014 no dispatch notification",
            "body": (
                "I placed order ORD-558102 eighteen days ago with an "
                "estimated dispatch window of 3\u20135 business days. The "
                "order status has remained on 'processing' for the entire "
                "period with no dispatch notification, no tracking number, "
                "and no communication from your team. I have sent two "
                "previous emails with no response, which is why I am "
                "escalating here. The order total was $215. At this point "
                "I need either a confirmed dispatch date with tracking "
                "information within 24 hours, or a full refund so I can "
                "source the item elsewhere."
            ),
            "customer_name": "Chioma Eze",
            "customer_email": "c.eze@creativeagency-example.ng",
            "previous_interactions": 2,
            "ground_truth_category": "SHIPPING",
            "ground_truth_priority": "HIGH",
            "ground_truth_team": "logistics_team",
            "ground_truth_resolution_hours": 8,
        },
        {
            "subject": "Wrong size received for order #ORD-229344 \u2014 need exchange",
            "body": (
                "I ordered a size large in item SKU-44821 but received a "
                "size medium in the delivery for order ORD-229344. The "
                "packing slip inside the box actually shows the correct "
                "size (large) but the item itself is a medium. I have "
                "photographs of both the packing slip and the item label "
                "confirming the discrepancy. I would like to arrange an "
                "exchange for the correct size. Please advise whether I "
                "need to return the incorrect item before the replacement "
                "is dispatched, and whether there is a prepaid return "
                "label available given that the error was on your end."
            ),
            "customer_name": "Emre Yilmaz",
            "customer_email": "e.yilmaz@retailclient-example.tr",
            "previous_interactions": 0,
            "ground_truth_category": "SHIPPING",
            "ground_truth_priority": "MEDIUM",
            "ground_truth_team": "logistics_team",
            "ground_truth_resolution_hours": 24,
        },
        {
            "subject": "Tracking number provided is invalid \u2014 carrier website shows 'not found'",
            "body": (
                "I received an order dispatch notification for ORD-881203 "
                "four days ago with tracking number 1Z999AA10123456784. "
                "When I enter this number on the carrier website it returns "
                "'tracking number not found'. I have waited 48 hours in "
                "case it was a pre-advice label not yet scanned, but the "
                "number still does not resolve. Could you confirm whether "
                "the item has actually been collected by the carrier or "
                "whether there was an error in the tracking number "
                "generation? I need to plan for the delivery as someone "
                "needs to be present to receive it."
            ),
            "customer_name": "Sunita Sharma",
            "customer_email": "s.sharma@homeoffice-example.in",
            "previous_interactions": 1,
            "ground_truth_category": "SHIPPING",
            "ground_truth_priority": "MEDIUM",
            "ground_truth_team": "logistics_team",
            "ground_truth_resolution_hours": 12,
        },
        {
            "subject": "Question about international delivery timelines to South Africa",
            "body": (
                "I am considering placing a larger order but before I "
                "do I would like to understand your international delivery "
                "timelines and costs to Johannesburg, South Africa. Your "
                "checkout page shows a delivery estimate of 7\u201314 business "
                "days for international economy but I have seen other "
                "customers report much longer actual delivery times in "
                "forums. Is the 7\u201314 business day estimate accurate for "
                "South Africa specifically, and does it include customs "
                "clearance time? I also want to understand your process "
                "if an item is held in customs \u2014 do you handle the "
                "clearance or does the responsibility pass to the "
                "recipient?"
            ),
            "customer_name": "Thabo Nkosi",
            "customer_email": "t.nkosi@importco-example.za",
            "previous_interactions": 0,
            "ground_truth_category": "SHIPPING",
            "ground_truth_priority": "LOW",
            "ground_truth_team": "logistics_team",
            "ground_truth_resolution_hours": 48,
        },
        # ── GENERAL CATEGORY (6 tickets) ─────────────────────────────────
        {
            "subject": "Request for custom data retention policy \u2014 compliance requirement",
            "body": (
                "We operate in a regulated industry and our compliance "
                "team requires that all customer data processed through "
                "your platform be deleted within 90 days of user profile "
                "closure. Your standard data retention policy states "
                "data is held for 180 days post-closure. We need written "
                "confirmation that a custom 90-day retention policy can "
                "be applied to our user profile (ACC-102933), or alternatively "
                "a Data Processing Agreement amendment that satisfies "
                "our regulatory requirements. This is blocking our "
                "procurement approval and has a deadline of 15 business "
                "days."
            ),
            "customer_name": "Miriam Goldstein",
            "customer_email": "m.goldstein@compliancefirst-example.il",
            "previous_interactions": 1,
            "ground_truth_category": "GENERAL",
            "ground_truth_priority": "HIGH",
            "ground_truth_team": "general_team",
            "ground_truth_resolution_hours": 24,
        },
        {
            "subject": "Enterprise plan inquiry \u2014 500 seat deployment with SSO and audit logs",
            "body": (
                "We are evaluating your platform for a 500-seat enterprise "
                "deployment and have several questions before we can "
                "proceed to a formal proposal. We require SAML 2.0 SSO, "
                "audit log export via API, a 99.9 percent uptime SLA in "
                "writing, and dedicated user profile management. Your public "
                "pricing page does not detail these features for the "
                "enterprise tier. Could you arrange a call with a "
                "solutions engineer within the next five business days? "
                "We are on a procurement timeline with a board approval "
                "deadline at the end of next month."
            ),
            "customer_name": "Jonathan Mbeki",
            "customer_email": "j.mbeki@enterprise-example.za",
            "previous_interactions": 0,
            "ground_truth_category": "GENERAL",
            "ground_truth_priority": "HIGH",
            "ground_truth_team": "general_team",
            "ground_truth_resolution_hours": 24,
        },
        {
            "subject": "Accessibility audit findings \u2014 keyboard navigation gaps in core workflows",
            "body": (
                "Our internal accessibility team has completed a WCAG 2.1 "
                "audit of your platform and identified several keyboard "
                "navigation gaps in core workflows. Specifically: the "
                "modal dialogs in the project creation flow are not "
                "keyboard-dismissible, the data table sorting controls "
                "lack focus indicators, and the notification dropdown "
                "does not trap focus correctly. We have 12 team members "
                "who rely on keyboard-only navigation and are currently "
                "unable to complete these workflows. Could you confirm "
                "whether these issues are on your accessibility roadmap "
                "and if so provide an approximate timeline for remediation?"
            ),
            "customer_name": "Aisha Mohammed",
            "customer_email": "a.mohammed@a11yteam-example.ae",
            "previous_interactions": 0,
            "ground_truth_category": "GENERAL",
            "ground_truth_priority": "MEDIUM",
            "ground_truth_team": "general_team",
            "ground_truth_resolution_hours": 48,
        },
        {
            "subject": "API documentation discrepancy \u2014 response schema differs from actual response",
            "body": (
                "The documentation for the /v2/reports endpoint states "
                "that the response includes a 'summary' object containing "
                "'total_count' and 'filtered_count' fields. The actual "
                "API response does not include this 'summary' object at "
                "all \u2014 only the 'data' array is returned. This discrepancy "
                "has caused our team to spend several days debugging what "
                "we initially thought was an integration error on our side. "
                "Could you confirm which is correct \u2014 the documentation "
                "or the implementation \u2014 and if the documentation is wrong, "
                "update it? Our integration depends on knowing the total "
                "record count for pagination."
            ),
            "customer_name": "Luca Bianchi",
            "customer_email": "l.bianchi@devteam-example.it",
            "previous_interactions": 1,
            "ground_truth_category": "GENERAL",
            "ground_truth_priority": "MEDIUM",
            "ground_truth_team": "general_team",
            "ground_truth_resolution_hours": 48,
        },
        {
            "subject": "Feature request: bulk export of all projects as a single ZIP archive",
            "body": (
                "Our team regularly needs to archive completed projects "
                "for regulatory record-keeping. Currently the only way "
                "to export project data is one project at a time through "
                "the UI, which for our typical archive of 40\u201360 projects "
                "takes approximately two hours of manual work. We would "
                "like to request a bulk export feature that allows "
                "selection of multiple projects and generates a single "
                "ZIP archive. I understand this may be a roadmap item "
                "rather than an immediate fix. Could you confirm whether "
                "this is being considered and whether there is a timeline, "
                "or alternatively whether the API provides a programmatic "
                "way to achieve this today?"
            ),
            "customer_name": "Hana Novakova",
            "customer_email": "h.novakova@archiveops-example.cz",
            "previous_interactions": 0,
            "ground_truth_category": "GENERAL",
            "ground_truth_priority": "LOW",
            "ground_truth_team": "general_team",
            "ground_truth_resolution_hours": 72,
        },
        {
            "subject": "Request for training session for new team of 15 users",
            "body": (
                "We have recently onboarded 15 new team members who will "
                "be using the platform as part of their daily workflows. "
                "None of them have prior experience with the system and "
                "we would like to arrange a structured training session "
                "covering the core features relevant to our use case "
                "(reporting, project management, and API integration). "
                "Our customer success manager is Jenna Park but she "
                "has been unresponsive to emails for the past two weeks. "
                "Could you either connect us with Jenna directly or "
                "arrange an alternative contact who can schedule a "
                "90-minute onboarding session within the next two weeks?"
            ),
            "customer_name": "Patrick Otieno",
            "customer_email": "p.otieno@growthco-example.ke",
            "previous_interactions": 2,
            "ground_truth_category": "GENERAL",
            "ground_truth_priority": "LOW",
            "ground_truth_team": "general_team",
            "ground_truth_resolution_hours": 72,
        },
    ]

    def fetch(self, n: int, seed: int = 42) -> list:
        """
        Return n LabeledTicket objects sampled from the 30 synthetic tickets.

        Uses the provided seed for deterministic ordering. If n > 30, cycles
        through the pool with shuffled order.

        Args:
            n: Number of tickets to return.
            seed: Random seed for deterministic sampling.

        Returns:
            List of exactly n LabeledTicket objects with label_source
            set to 'realistic_synthetic'.
        """
        from server.data.fetcher import LabeledTicket

        rng = random.Random(seed)
        now_iso = "2025-06-01T00:00:00Z"

        pool = list(self.TICKETS)
        rng.shuffle(pool)

        result: list = []
        idx = 0

        while len(result) < n:
            if idx >= len(pool):
                # Re-shuffle and cycle
                rng.shuffle(pool)
                idx = 0

            raw = pool[idx]
            idx += 1

            ticket = Ticket(
                ticket_id=f"RS-{len(result) + 1:03d}",
                subject=raw["subject"],
                body=raw["body"],
                customer_name=raw["customer_name"],
                customer_email=raw["customer_email"],
                created_at=now_iso,
                attachments=[],
                previous_interactions=raw["previous_interactions"],
            )

            ground_truth = {
                "category": raw["ground_truth_category"],
                "priority": raw["ground_truth_priority"],
                "team": raw["ground_truth_team"],
                "resolution_hours": raw["ground_truth_resolution_hours"],
            }

            result.append(
                LabeledTicket(
                    ticket=ticket,
                    ground_truth=ground_truth,
                    label_source="realistic_synthetic",
                )
            )

        return result
