# Dataset — Artemis (Lead-Qualification)

Illustrative reference dataset for fine-tuning a B2B inbound lead-qualification agent. The full set is not distributed; [`sample.jsonl`](sample.jsonl) contains 12 representative rows that show the exact schema and label style used in training and eval.

## Schema

One JSON object per line:

```json
{
  "conversation": "<the inbound lead messages, as text>",
  "qualification": {
    "qualified": true,
    "reason": "<one sentence, grounded in the lead's own words>",
    "next_action": "book_demo | nurture | route_to_sales | disqualify | request_info"
  },
  "reply": "<a ready-to-send SDR reply for a human to review>"
}
```

| Field | Type | Notes |
|---|---|---|
| `conversation` | string | Raw inbound text (chat / email / web-form). PII redacted or surrogated. |
| `qualification.qualified` | bool | Analyst label. Positive class for F1. |
| `qualification.reason` | string | One sentence; must cite stated business signals (budget, timeline, pain, authority), not demographics. |
| `qualification.next_action` | enum | `book_demo`, `nurture`, `route_to_sales`, `disqualify`, `request_info`. |
| `reply` | string | Gold SDR reply; the human-in-the-loop draft target. |

## Size and composition (reference)

| | Count |
|---|---|
| Conversations after cleaning/dedup | ~12,400 |
| Train | 11,000 |
| Validation | 400 |
| Held-out replay test | 1,000 |

- **Source mix:** ~70% redacted real inbound conversations; ~30% synthetic, generated to cover edge cases (vague budget, multi-stakeholder, competitor mentions, spam-like inbound, urgent gaps).
- **Class balance (test):** 420 qualified / 580 not qualified — see [`../eval/results.md`](../eval/results.md).
- **Labeling:** each row double-reviewed by sales analysts; disagreements resolved by a senior analyst.

## Splits

Splits are **conversation-disjoint** and **time-ordered**: the test window is the most recent, so the model is evaluated on conversations that postdate everything it trained on. This guards against leakage and approximates a real "going forward" deployment.

## Privacy

Names, company identifiers, contact details, and account-specific dollar figures are redacted or replaced with surrogates before training. The sample rows here are fictional but constructed to be representative of real inbound patterns.

## Intended use note

This data encodes a specific qualification policy. If your definition of "qualified" differs, re-label against your own policy and re-tune — the model inherits the labeling team's judgment. Do not use the dataset or model to infer or act on protected attributes.
