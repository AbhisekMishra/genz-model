"""Programmatically convert already-paired HF datasets (plain-english /
genz-slang text pairs) directly into slang_translation training examples.

Pure data transformation -- no LLM calls, no tokens spent reading rows.
Run after scripts/download_hf_datasets.py.
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "data" / "synthetic" / "slang_translation_from_raw.jsonl"

SYSTEM_PROMPT = "Rewrite the user's text in genz slang. Keep the meaning the same."

# same exclusion list used when briefing the synthetic-authoring subagents
BANNED_SUBSTRINGS = ["gyat", "mog", "looksmax", "grindset"]

MIN_LEN = 8
MAX_LEN = 220
SEED = 42

# (filename, normal-text key, slang-text key, max examples to keep from this source)
SOURCES = [
    ("Programmer-RD-AI__genz-slang-pairs-1k.jsonl", "normal", "gen_z", 200),
    ("thesherrycode__gen-z-slangs-translation.jsonl", "Plain English", "Gen-Z Slang", None),
]


def clean(text: str) -> str:
    return " ".join(text.strip().split())


def is_acceptable(normal: str, slang: str) -> bool:
    if not normal or not slang:
        return False
    if not (MIN_LEN <= len(normal) <= MAX_LEN):
        return False
    if not (MIN_LEN <= len(slang) <= MAX_LEN):
        return False
    combined = (normal + " " + slang).lower()
    return not any(bad in combined for bad in BANNED_SUBSTRINGS)


def load_pairs(path: Path, normal_key: str, slang_key: str) -> list[tuple[str, str]]:
    pairs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            normal = clean(obj.get(normal_key, ""))
            slang = clean(obj.get(slang_key, ""))
            if is_acceptable(normal, slang):
                pairs.append((normal, slang))
    return pairs


def main() -> None:
    rng = random.Random(SEED)
    seen_normal = set()
    examples = []

    for filename, normal_key, slang_key, cap in SOURCES:
        path = RAW_DIR / filename
        if not path.exists():
            print(f"[skip] {filename} not found (run download_hf_datasets.py first)")
            continue

        pairs = load_pairs(path, normal_key, slang_key)
        # dedup within-source and against already-kept examples
        deduped = []
        for normal, slang in pairs:
            key = normal.lower()
            if key in seen_normal:
                continue
            seen_normal.add(key)
            deduped.append((normal, slang))

        if cap is not None and len(deduped) > cap:
            rng.shuffle(deduped)
            deduped = deduped[:cap]

        for normal, slang in deduped:
            examples.append(
                {
                    "category": "slang_translation",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": normal},
                        {"role": "assistant", "content": slang},
                    ],
                }
            )
        print(f"[ok] {filename}: {len(pairs)} passed filters -> {len(deduped)} kept (cap={cap})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(examples)} examples -> {OUT_PATH}")


if __name__ == "__main__":
    main()
