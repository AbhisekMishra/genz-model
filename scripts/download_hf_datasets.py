"""Download HF grounding datasets for genz slang vocabulary into data/raw/.

These are used as reference material (real slang terms/definitions/examples)
to keep the Claude-authored synthetic data grounded in authentic vocabulary,
not as the primary training set.
"""

import json
from pathlib import Path

from datasets import load_dataset

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# dataset_id -> config/split kwargs (empty dict = defaults)
DATASETS = {
    "MLBtrio/genz-slang-dataset": {},
    "Programmer-RD-AI/genz-slang-pairs-1k": {},
    "thesherrycode/gen-z-slangs-translation": {},
    "nikesh66/Slang-Dataset": {},
}


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_id, kwargs in DATASETS.items():
        safe_name = dataset_id.replace("/", "__")
        out_path = RAW_DIR / f"{safe_name}.jsonl"
        try:
            ds = load_dataset(dataset_id, **kwargs)
            split = ds[list(ds.keys())[0]]
            with out_path.open("w", encoding="utf-8") as f:
                for row in split:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[ok] {dataset_id}: {len(split)} rows -> {out_path}")
        except Exception as e:  # dataset may be renamed/removed/have a different schema
            print(f"[skip] {dataset_id}: {e}")


if __name__ == "__main__":
    main()
