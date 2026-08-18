"""Merge the trained LoRA adapter into the base model weights and save a
standalone merged model to outputs/merged_model/ (full precision, no
quantization) so it can be loaded with a plain `from_pretrained` call.
"""

from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "train_config.yaml"


def main() -> None:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    adapter_dir = ROOT / cfg["adapter_dir"]
    merged_dir = ROOT / "outputs" / "merged_model"
    merged_dir.mkdir(parents=True, exist_ok=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])

    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model = model.merge_and_unload()

    model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    print(f"Merged model saved to {merged_dir}")


if __name__ == "__main__":
    main()
