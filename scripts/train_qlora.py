"""QLoRA fine-tune of the base model on the genz slang dataset.

Usage:
    python scripts/train_qlora.py --config configs/train_config.yaml
    python scripts/train_qlora.py --config configs/train_config.yaml --max_steps 100   # pilot run
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent.parent


def load_yaml_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_and_tokenizer(cfg: dict):
    compute_dtype = getattr(torch, cfg["bnb_4bit_compute_dtype"])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False
    return model, tokenizer


def apply_lora(model, lora_config_path: str):
    with open(lora_config_path, encoding="utf-8") as f:
        lora_cfg = json.load(f)
    peft_config = LoraConfig(**lora_cfg)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def formatting_func(tokenizer):
    def _format(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    return _format


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_steps", type=int, default=-1, help="limit steps for a pilot run")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    model, tokenizer = build_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg["lora_config"])

    train_ds = load_dataset("json", data_files=str(ROOT / cfg["train_file"]), split="train")
    val_ds = load_dataset("json", data_files=str(ROOT / cfg["val_file"]), split="train")

    sft_config = SFTConfig(
        output_dir=str(ROOT / cfg["output_dir"]),
        max_length=cfg["max_seq_length"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_steps=cfg["warmup_steps"],
        optim=cfg["optim"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        logging_steps=cfg["logging_steps"],
        eval_strategy=cfg["eval_strategy"],
        eval_steps=cfg["eval_steps"],
        save_strategy=cfg["save_strategy"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        bf16=cfg["bf16"],
        seed=cfg["seed"],
        max_steps=args.max_steps,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        formatting_func=formatting_func(tokenizer),
        processing_class=tokenizer,
    )

    trainer.train()

    adapter_dir = ROOT / cfg["adapter_dir"]
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Saved LoRA adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
