"""Merge synthetic per-category JSONL batches into a deduped, stratified
train/val split written to data/processed/.

Expects each line in data/synthetic/*.jsonl to look like:
    {"category": "casual_chat", "messages": [{"role": ..., "content": ...}, ...]}
"""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

SYNTHETIC_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
VAL_FRACTION = 0.10
SEED = 42


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def example_hash(example: dict) -> str:
    joined = normalize(" ".join(m["content"] for m in example["messages"]))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def load_all() -> list[dict]:
    examples = []
    for path in sorted(SYNTHETIC_DIR.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{line_no} invalid JSON: {e}") from e
                if "messages" not in obj or "category" not in obj:
                    raise ValueError(f"{path}:{line_no} missing 'messages' or 'category'")
                examples.append(obj)
    return examples


def dedup(examples: list[dict]) -> tuple[list[dict], int]:
    seen = set()
    out = []
    dup_count = 0
    for ex in examples:
        h = example_hash(ex)
        if h in seen:
            dup_count += 1
            continue
        seen.add(h)
        out.append(ex)
    return out, dup_count


def stratified_split(examples: list[dict]) -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)
    by_category = defaultdict(list)
    for ex in examples:
        by_category[ex["category"]].append(ex)

    train, val = [], []
    for category, items in by_category.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * VAL_FRACTION))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_jsonl(path: Path, examples: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def print_stats(examples: list[dict], label: str) -> None:
    by_category = defaultdict(int)
    lengths = []
    for ex in examples:
        by_category[ex["category"]] += 1
        lengths.append(sum(len(m["content"]) for m in ex["messages"]))
    lengths.sort()
    n = len(lengths)
    print(f"\n[{label}] total={n}")
    for cat, count in sorted(by_category.items()):
        print(f"  {cat}: {count}")
    if n:
        print(
            f"  char-length: min={lengths[0]} p50={lengths[n // 2]} "
            f"p95={lengths[int(n * 0.95)]} max={lengths[-1]}"
        )


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_all()
    if not raw:
        print(f"No synthetic data found in {SYNTHETIC_DIR}")
        return

    deduped, dup_count = dedup(raw)
    print(f"Loaded {len(raw)} examples, removed {dup_count} exact duplicates, {len(deduped)} remain")

    train, val = stratified_split(deduped)
    write_jsonl(PROCESSED_DIR / "train.jsonl", train)
    write_jsonl(PROCESSED_DIR / "val.jsonl", val)

    print_stats(train, "train")
    print_stats(val, "val")
    print(f"\nWrote {PROCESSED_DIR / 'train.jsonl'} and {PROCESSED_DIR / 'val.jsonl'}")


if __name__ == "__main__":
    main()
