"""Build a compact, deduped term-reference table from the MLBtrio slang
dataset for use as grounding material for future data-authoring passes.

The point of this script: the raw dataset (1,779 rows, many duplicate or
near-duplicate terms) is too large to read into an LLM context cheaply.
This compresses it locally (pure Python, no LLM calls) into a short
markdown table -- small enough to actually read or hand to an authoring
subagent.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "MLBtrio__genz-slang-dataset.jsonl"
OUT_PATH = ROOT / "data" / "vocab_reference.md"

MAX_DESC_LEN = 140


def main() -> None:
    if not RAW_PATH.exists():
        print(f"{RAW_PATH} not found (run download_hf_datasets.py first)")
        return

    seen: dict[str, tuple[str, str]] = {}
    total_rows = 0
    with RAW_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_rows += 1
            obj = json.loads(line)
            term = obj.get("Slang", "").strip()
            desc = obj.get("Description", "").strip()
            if not term or not desc:
                continue
            key = term.lower()
            if key in seen:
                continue  # keep first occurrence only
            if len(desc) > MAX_DESC_LEN:
                desc = desc[:MAX_DESC_LEN].rsplit(" ", 1)[0] + "..."
            seen[key] = (term, desc)

    lines = [f"- **{term}**: {desc}" for term, desc in sorted(seen.values(), key=lambda x: x[0].lower())]
    OUT_PATH.write_text("# GenZ vocabulary reference\n\n" + "\n".join(lines) + "\n", encoding="utf-8")

    print(f"{total_rows} raw rows -> {len(seen)} unique terms -> {OUT_PATH}")


if __name__ == "__main__":
    main()
