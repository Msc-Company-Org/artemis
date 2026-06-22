# Model Card — Artemis (Lead-Qualification Agent)

Artemis is a QLoRA fine-tune of `mistralai/Mistral-7B-Instruct-v0.3` for **B2B inbound lead qualification**. Given a raw inbound sales conversation, it returns a structured qualification decision, a grounded reason, a recommended next action, and a drafted SDR reply.

- **Developed by:** MSC Labs
- **Model type:** Decoder-only causal LM with a LoRA adapter (PEFT)
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Language:** English
- **License:** Apache-2.0 (code and adapter config). Reference weights/data are illustrative and not distributed.
- **Finetuned from:** Mistral-7B-Instruct-v0.3

## Intended use

**Primary use.** Front-of-funnel triage for inbound B2B sales conversations: decide whether a lead is qualified, explain why in one sentence grounded in the lead's own words, recommend a next action (`book_demo`, `nurture`, `route_to_sales`, `disqualify`, `request_info`), and draft a first reply for an SDR to review and send.

**Intended users.** Revenue / RevOps teams who want consistent, low-cost qualification at the top of the funnel and a human in the loop on every send.

**Out of scope.**
- Autonomous sending of replies without human review. Artemis drafts; a person sends.
- Final purchase, pricing, or contract decisions.
- High-stakes or regulated outreach (e.g., financial advice, healthcare eligibility) without domain review.
- Non-English conversations (not trained or evaluated for them).
- Demographic, creditworthiness, or any protected-attribute inference. The model is not designed or evaluated for these and must not be used for them.

## Training data

Illustrative reference dataset of B2B inbound sales conversations, each labeled by sales analysts. A representative sample of the row schema is in [`data/sample.jsonl`](data/sample.jsonl); see [`data/README.md`](data/README.md) for the full description.

- **Size (reference):** ~12,400 conversations after cleaning and dedup.
- **Composition:** ~70% redacted real inbound chats/emails/web-form threads; ~30% synthetic conversations generated to cover edge cases (vague budget, multi-stakeholder, competitor mentions, spam-like inbound).
- **Splits:** 11,000 train / 400 validation / 1,000 held-out replay test. Splits are conversation-disjoint and time-ordered (test is the most recent window) to avoid leakage.
- **Labels:** `qualified` (bool), `reason` (free text), `next_action` (enum), and a gold SDR `reply`. Each label double-reviewed; disagreements resolved by a senior analyst.
- **Row schema:** `{"conversation": ..., "qualification": {"qualified", "reason", "next_action"}, "reply": ...}`.

PII (names, company identifiers, contact details, dollar figures tied to a real account) is redacted or surrogated before training.

## Training procedure

**Method:** QLoRA — 4-bit NF4 base, LoRA adapters on attention and MLP projections, bf16 compute.

**2-stage protocol (MSC Labs standard):**
1. **Smoke test.** 200 steps on a 1% data slice to validate the data schema, loss curve, tokenization, and prompt formatting. Cheap, fast, catches config errors before paying for the full run.
2. **Full run.** 3 epochs on the full train split once the smoke test is clean.

**Prompt format.** Mistral instruction format (`<s>[INST] ... [/INST]`). The system block fixes the JSON output contract; the response is the JSON object `{qualification, reply}`. Loss is masked on the prompt; only the completion contributes.

**Key hyperparameters** (see [`training/config.yaml`](training/config.yaml)):
- LoRA rank 16, alpha 32, dropout 0.05, on `q/k/v/o/gate/up/down` projections.
- Effective batch size 32 (per-device 4 × grad-accum 8), max seq len 2048.
- LR 2e-4, cosine schedule, 3% warmup, paged AdamW 8-bit.
- 3 epochs, bf16, gradient checkpointing.

**Compute (reference):** 1× A100 80GB, ~6.5 hours wall-clock for the full run.

## Evaluation

See [`eval/results.md`](eval/results.md) and [`eval/results.json`](eval/results.json) for full methodology and numbers.

- **Setup:** Offline replay on a 1,000-conversation held-out set. Each conversation is scored against analyst ground truth. GPT-4o is run on identical prompts as the baseline.
- **Headline (illustrative reference results from the MSC Labs eval harness):**
  - Qualification F1: **0.93** (GPT-4o 0.86).
  - Qualified-lead rate (positive-class recall): **85.0%** vs. GPT-4o 71.4% → **+19% relative**.
  - Next-action accuracy: **0.88** (GPT-4o 0.79).
  - Reply human-preference win rate vs. GPT-4o: **61%**.
  - Cost: **$0.20 / 1k requests** vs. $5.00 → **25× cheaper**. p50 latency 540 ms vs. 1,180 ms.

## Limitations

- **English / B2B-inbound only.** Performance outside this distribution is unknown.
- **Replay eval, not live A/B.** Offline replay approximates but does not equal production lift; treat the numbers as directional until validated in a live test.
- **Label-bounded.** The model inherits the qualification policy of the labeling team. If your definition of "qualified" differs, re-label and re-tune.
- **JSON contract can break** on adversarial or extremely long inputs; downstream code should validate and fall back gracefully.
- **Reply drafts need human review.** They can be confidently wrong about facts (integrations, pricing) the model was not grounded on.

## Bias, risks, and recommendations

- **Selection bias.** A qualification model can systematically advantage or disadvantage segments present in the training labels (company size, region, industry). Audit qualified-rate parity across segments before relying on it for routing.
- **No protected attributes.** The model must not be used to infer or act on protected characteristics. Reasons should cite stated business signals (budget, timeline, pain, authority), not demographics.
- **Human in the loop.** Keep a person on every reply send and on disqualification decisions; log overrides and feed them back into the next training round.
- **Drift.** Inbound language and offers change. Re-evaluate quarterly and re-tune when qualified-rate or F1 drifts beyond your threshold.

---

> Reference model by **MSC Labs** — done-for-you custom model training.
> Want this for your task? → Book a free model audit: https://labs.msccompany.com.br/assessment
> Numbers are illustrative reference results from our standard eval harness.
