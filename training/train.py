"""
Artemis — QLoRA fine-tuning for B2B lead qualification.

Fine-tunes Mistral-7B-Instruct-v0.3 to read an inbound sales conversation and
emit JSON: {qualification: {qualified, reason, next_action}, reply}.

Follows the MSC Labs 2-stage protocol:
    1. smoke test  — 200 steps on 1% of data (validate config cheaply)
    2. full run    — 3 epochs on the full train split

Usage:
    python training/train.py --config training/config.yaml --stage smoke_test
    python training/train.py --config training/config.yaml --stage full_run

Illustrative reference script. Looks runnable; weights/data are not distributed.
"""

import argparse
import json

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

SYSTEM_PROMPT = (
    "You are a B2B inbound sales qualification agent. Given the lead conversation, "
    "return JSON with keys: qualification {qualified, reason, next_action} and reply. "
    "Ground the reason in the lead's own words; pick next_action from "
    "[book_demo, nurture, route_to_sales, disqualify, request_info]."
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_example(row: dict) -> dict:
    """Render one row into the Mistral instruction format.

    Completion is the JSON object the model must learn to produce. The prompt is
    masked at loss time (mask_prompt_loss=true), so only the completion trains.
    """
    completion = json.dumps(
        {"qualification": row["qualification"], "reply": row["reply"]},
        ensure_ascii=False,
    )
    text = (
        f"<s>[INST] {SYSTEM_PROMPT}\n\n"
        f"Conversation:\n{row['conversation']} [/INST] {completion}</s>"
    )
    return {"text": text}


def build_model_and_tokenizer(cfg: dict):
    q = cfg["quantization"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
    )

    base = cfg["model"]["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=bnb,
        device_map="auto",
        attn_implementation=cfg["model"]["attn_implementation"],
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora = cfg["lora"]
    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["lora_alpha"],
        lora_dropout=lora["lora_dropout"],
        bias=lora["bias"],
        task_type=lora["task_type"],
        target_modules=lora["target_modules"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--stage", choices=["smoke_test", "full_run"], default="full_run")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg, train_cfg = cfg["data"], cfg["training"]

    dataset = load_dataset("json", data_files=data_cfg["train_file"], split="train")

    stage = cfg["stages"][args.stage]
    if args.stage == "smoke_test":
        n = max(1, int(len(dataset) * stage["data_fraction"]))
        dataset = dataset.select(range(n))
        max_steps, num_epochs = stage["max_steps"], 1
        print(f"[smoke_test] {n} rows, {max_steps} steps — validating config before the full run.")
    else:
        max_steps, num_epochs = -1, stage["num_train_epochs"]
        print(f"[full_run] {len(dataset)} rows, {num_epochs} epochs.")

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    model, tokenizer = build_model_and_tokenizer(cfg)

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        optim=train_cfg["optim"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        max_grad_norm=train_cfg["max_grad_norm"],
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        seed=train_cfg["seed"],
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=data_cfg["max_seq_len"],
        packing=False,
    )

    trainer.train()
    trainer.save_model(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])
    print(f"Saved adapter to {train_cfg['output_dir']}")


if __name__ == "__main__":
    main()
