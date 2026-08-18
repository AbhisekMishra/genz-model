"""Upload the LoRA adapter and the merged full-weights model to two
Hugging Face Hub repos.

Requires a write-scope HF token, e.g.:
    huggingface-cli login
or set HF_TOKEN in the environment.

Usage:
    python scripts/push_to_hub.py --username YOUR_HF_USERNAME [--private]
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent


def push_folder(api: HfApi, folder: Path, repo_id: str, private: bool) -> None:
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="model")
    print(f"Pushed {folder} -> https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True, help="your HF username or org")
    parser.add_argument("--model-name", default="genz-slang-model")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()

    adapter_dir = ROOT / "outputs" / "adapter"
    merged_dir = ROOT / "outputs" / "merged_model"

    # copy the shared model card into both output dirs so each repo has one
    model_card = (ROOT / "model_card.md").read_text(encoding="utf-8")
    (adapter_dir / "README.md").write_text(model_card, encoding="utf-8")
    (merged_dir / "README.md").write_text(model_card, encoding="utf-8")

    push_folder(api, adapter_dir, f"{args.username}/{args.model_name}-lora", args.private)
    push_folder(api, merged_dir, f"{args.username}/{args.model_name}", args.private)


if __name__ == "__main__":
    main()
