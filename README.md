# genz-model

[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-abhisekm%2Fgenz--slang--model-yellow)](https://huggingface.co/abhisekm/genz-slang-model)

A small open-source LLM fine-tuned to speak in genz internet slang, published to Hugging Face.

Two capabilities in one model, switched via system prompt:
1. **Persona chat** — free-form conversation in a genz voice.
2. **Style transfer** — rewrite normal text into genz slang.

## Approach

- Base model: [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Apache 2.0), fine-tuned with QLoRA (4-bit).
- Training data (~4,600 examples): hand-authored + parallel-generated synthetic examples (persona chat, style-transfer pairs, roleplay, slang Q&A, edge cases) plus real Hugging Face slang datasets used two ways — `MLBtrio/genz-slang-dataset` compressed into a deduped term-reference file for grounding, and the already-paired `Programmer-RD-AI/genz-slang-pairs-1k` / `thesherrycode/gen-z-slangs-translation` datasets converted programmatically (filtered, deduped, capped — no LLM calls) directly into training examples.
- Trained locally on an 8GB VRAM GPU, ~20-25 min per full run.

See `configs/` for training hyperparameters and `model_card.md` for the published model documentation. For a full chronological write-up of every decision, bug, and debugging round — written as a learning resource — see [`LEARNINGS.md`](./LEARNINGS.md).

## Repo layout

```
configs/            LoRA + training config
data/
  raw/               downloaded HF grounding datasets (gitignored)
  synthetic/          authored + programmatically-converted training examples (per category, JSONL)
  processed/           merged/deduped train.jsonl + val.jsonl
  vocab_reference.md   deduped term-reference table built from MLBtrio/genz-slang-dataset
scripts/             data build, training, eval, upload scripts
outputs/             checkpoints / adapter / merged model (gitignored)
model_card.md        Hugging Face model card
```

## Pipeline

```bash
python scripts/download_hf_datasets.py     # pull grounding datasets
python scripts/convert_pair_datasets.py    # convert already-paired HF datasets -> training examples (no LLM calls)
python scripts/build_vocab_reference.py    # compress MLBtrio dataset into a queryable term-reference file
python scripts/build_dataset.py            # merge, dedup, split synthetic data
python scripts/train_qlora.py --config configs/train_config.yaml
python scripts/eval_compare.py             # base vs fine-tuned qualitative comparison
python scripts/merge_lora.py               # merge LoRA into base weights
python scripts/push_to_hub.py              # upload adapter + merged repos
```
