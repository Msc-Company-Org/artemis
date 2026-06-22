# Artemis — Sales / Lead-Qualification Agent

**Qualify inbound leads, score fit, and draft the next reply — at 1/25th the cost of GPT-4o.**

Artemis is a fine-tuned **Mistral-7B-Instruct-v0.3** that reads a raw inbound sales conversation and returns three things: a qualification decision (`qualified` / `not qualified`), a short reason grounded in the message text, a recommended next action, and a ready-to-send SDR reply. It is built for the front of the funnel, where speed and consistency matter more than open-ended chat.

## What and why

Frontier APIs qualify leads well but cost too much to run on every inbound message, and they drift in tone across a team. The MSC Labs approach: take a focused, labeled slice of your own sales conversations, fine-tune a small open model with QLoRA, and pin the behavior. The result runs cheaply on a single GPU and stays consistent.

- **Task:** B2B inbound lead qualification + next-action + reply drafting.
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`.
- **Method:** QLoRA (4-bit), 2-stage protocol (smoke test → full run).
- **Eval:** offline replay against historical SDR decisions, plus F1 on human qualification labels.

## Results

Illustrative reference results from the MSC Labs eval harness. Baseline is **GPT-4o** on the same prompts and the same held-out replay set (1,000 conversations).

| Metric | GPT-4o (baseline) | Artemis (tuned 7B) | Delta |
|---|---|---|---|
| Qualified-lead rate (vs. ground-truth) | 71.4% | 85.0% | **+19.0% relative** |
| Qualification F1 | 0.86 | **0.93** | +0.07 |
| Next-action accuracy | 0.79 | 0.88 | +0.09 |
| Reply human-pref win rate | — | 61% | vs. GPT-4o |
| **$ / 1k requests** | $5.00 | **$0.20** | **25× cheaper** |
| p50 latency | 1,180 ms | 540 ms | 2.2× faster |

"Qualified-lead rate" is the share of truly-qualified leads the agent correctly routes forward (recall on the positive class), measured against analyst labels. Full methodology in [`eval/results.md`](eval/results.md).

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, json

BASE = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER = "MSC-Labs-AI/artemis-lead-qual"  # illustrative adapter id

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER)

SYSTEM = (
    "You are a B2B inbound sales qualification agent. Given the lead conversation, "
    "return JSON with keys: qualification {qualified, reason, next_action} and reply."
)

conversation = (
    "Lead: Hi, we're a 40-person logistics SaaS. Our support team is drowning in tickets. "
    "Do you integrate with Zendesk? Budget is around $2k/mo, need something live this quarter."
)

prompt = f"<s>[INST] {SYSTEM}\n\nConversation:\n{conversation} [/INST]"
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=512, temperature=0.2)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

Expected output (shape):

```json
{
  "qualification": {
    "qualified": true,
    "reason": "Named budget ($2k/mo), clear pain (ticket volume), and a buying timeline (this quarter).",
    "next_action": "book_demo"
  },
  "reply": "Thanks for reaching out! Yes — we have a native Zendesk integration. Given your timeline, I'd suggest a 20-minute demo this week. Does Thursday 2pm ET work?"
}
```

## Files

```
README.md            # this model card
MODEL_CARD.md        # formal HF-style card
training/config.yaml # QLoRA hyperparameters
training/train.py    # QLoRA fine-tuning script
eval/results.md      # methodology, results, cost, latency
eval/results.json    # machine-readable metrics
data/sample.jsonl    # sample inbound conversations + labels + replies
data/README.md       # dataset description
```

## License

Apache-2.0 for code (see [`LICENSE`](LICENSE) terms in the spec). Reference weights and datasets are illustrative and not distributed.

---

> Reference model by **MSC Labs** — done-for-you custom model training.
> Want this for your task? → Book a free model audit: https://labs.msccompany.com.br/assessment
> Numbers are illustrative reference results from our standard eval harness.
