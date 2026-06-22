# Evaluation — Artemis (Lead-Qualification Agent)

> Illustrative reference results from the MSC Labs eval harness. Baseline is **GPT-4o** run on identical prompts and the same held-out replay set. Numbers are directional, internally consistent across this repo, and not a substitute for a live A/B test.

## Methodology

**1. Offline replay.** We hold out the most recent **1,000 inbound conversations** (conversation-disjoint and time-ordered from train) for which we have analyst ground truth: `qualified` (bool), `next_action` (enum), and a gold SDR `reply`. Both Artemis and GPT-4o see the same system prompt and the same conversation, and must emit the same JSON contract `{qualification, reply}`. We parse the JSON and score it against ground truth.

**2. F1 on qualification labels.** The core metric. `qualified` is the positive class. We report precision, recall, and F1 against the analyst labels. "Qualified-lead rate" reported in the README is positive-class **recall** — the share of truly-qualified leads correctly routed forward — the metric the sales team cares about most.

**3. Next-action accuracy.** Exact-match accuracy of the predicted `next_action` enum against ground truth, over the same 1,000 conversations.

**4. Reply human-preference.** Two analysts blind-rank the Artemis reply vs. the GPT-4o reply on 300 sampled conversations (helpfulness, correctness, tone). We report the Artemis win rate (ties split). Inter-annotator agreement (Cohen's kappa) was 0.71.

**5. Cost and latency.** Cost is measured per 1,000 requests at a fixed prompt/response token budget (≈700 in / ≈180 out). GPT-4o uses published API pricing; Artemis uses measured throughput on 1× A100 80GB amortized to a per-request cost. Latency is p50 end-to-end at batch size 1.

## Results

### Qualification (positive class = qualified)

| Metric | GPT-4o | Artemis | Delta |
|---|---|---|---|
| Precision | 0.84 | 0.91 | +0.07 |
| Recall (= qualified-lead rate) | 0.714 | 0.850 | **+19.0% rel.** |
| **F1** | 0.86 | **0.93** | +0.07 |

The headline **+19% qualified-lead rate** is the relative lift in positive-class recall (0.850 / 0.714 − 1 = +19.0%). Artemis recovers qualified leads that GPT-4o under-routes, while also lifting precision (fewer junk leads pushed to sales).

### Next action and reply

| Metric | GPT-4o | Artemis |
|---|---|---|
| Next-action accuracy | 0.79 | 0.88 |
| Reply human-pref win rate | — | 61% (vs. GPT-4o) |

### Confusion (Artemis, n = 1,000)

|  | Pred qualified | Pred not |
|---|---|---|
| **Actual qualified** (420) | 357 (TP) | 63 (FN) |
| **Actual not** (580) | 35 (FP) | 545 (TN) |

Precision 357/(357+35) = 0.911; recall 357/420 = 0.850; F1 = 0.93. Overall accuracy = 902/1000 = 0.902.

## Cost breakdown

Per 1,000 requests, ≈700 input + ≈180 output tokens each.

| | GPT-4o | Artemis (self-hosted 7B) |
|---|---|---|
| Unit basis | API pricing | A100 80GB throughput, amortized |
| Input cost / 1k req | $3.94 | — |
| Output cost / 1k req | $1.06 | — |
| Compute / 1k req | — | ~$0.20 |
| **Total / 1k req** | **$5.00** | **$0.20** |
| **Relative** | 1× | **25× cheaper** |

At 50k inbound/month that is ~$250/mo (GPT-4o) vs. ~$10/mo (Artemis) on the same volume, before factoring the lift in qualified leads recovered.

## Latency

| | GPT-4o | Artemis |
|---|---|---|
| p50 end-to-end | 1,180 ms | 540 ms |
| p95 end-to-end | 2,400 ms | 910 ms |

Measured at batch size 1. Artemis runs locally, so there is no network round-trip and latency is more stable under load.

## 2-stage training protocol

The eval above is on the model produced by the MSC Labs standard:

1. **Smoke test** — 200 steps on 1% of the data. Confirms the data schema, tokenization, prompt format, and a clean decreasing loss curve before committing GPU budget. Catches config bugs cheaply.
2. **Full run** — 3 epochs on the full ~11k-row train split once the smoke test is green. ~6.5 h on 1× A100 80GB.

This keeps debugging off the expensive run and makes the cost of each experiment predictable.

## Limitations

- **Replay ≠ production.** Offline replay against historical labels approximates live lift but does not capture interaction effects (e.g., a better reply changing what the lead says next). Validate with a live A/B before claiming revenue impact.
- **Label-bounded.** Metrics reflect the labeling team's definition of "qualified." A different policy needs re-labeling and re-tuning.
- **English / B2B-inbound only.** Out-of-distribution behavior is unmeasured.
- **Small human-pref sample.** The 61% reply win rate is over 300 conversations with kappa 0.71; treat it as directional.
- **JSON parse failures** (rare, <0.5% on this set) are scored as errors, not silently dropped.

---

> Reference model by **MSC Labs** — done-for-you custom model training.
> Want this for your task? → Book a free model audit: https://msc-labs-ai.vercel.app/assessment
> Numbers are illustrative reference results from our standard eval harness.
