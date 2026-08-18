"""Qualitative side-by-side comparison of base vs fine-tuned model outputs
over a fixed held-out prompt set (data/eval_prompts.json). No automatic
metric is computed -- this is a human-review quality gate, printed to stdout
and also written to outputs/eval_comparison.md.
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "train_config.yaml"
PROMPTS_PATH = ROOT / "data" / "eval_prompts.json"
OUT_PATH = ROOT / "outputs" / "eval_comparison.md"

MAX_NEW_TOKENS = 200


def generate(model, tokenizer, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = output[0][input_ids.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(PROMPTS_PATH, encoding="utf-8") as f:
        prompts = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map="auto"
    )

    print("Attaching LoRA adapter...")
    # PeftModel.from_pretrained wraps base_model in place; use disable_adapter()
    # to still get pure base-model generations from the same loaded weights.
    model = PeftModel.from_pretrained(base_model, str(ROOT / cfg["adapter_dir"]))

    lines = ["# Base vs Fine-tuned comparison\n"]
    for i, p in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] ({p['type']}) {p['user'][:60]}")
        with model.disable_adapter():
            base_out = generate(model, tokenizer, p["system"], p["user"])
        ft_out = generate(model, tokenizer, p["system"], p["user"])

        block = (
            f"## [{p['type']}] {p['user']}\n\n"
            f"**system**: {p['system']}\n\n"
            f"**base**:\n{base_out}\n\n"
            f"**fine-tuned**:\n{ft_out}\n"
        )
        lines.append(block)
        print(f"  base:       {base_out[:120]}")
        print(f"  fine-tuned: {ft_out[:120]}\n")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n---\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
