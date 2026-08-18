# genz-model

[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-abhisekm%2Fgenz--slang--model-yellow)](https://huggingface.co/abhisekm/genz-slang-model)

A small open-source LLM fine-tuned to speak in genz internet slang, published to Hugging Face.

Two capabilities in one model, switched via system prompt:
1. **Persona chat** — free-form conversation in a genz voice.
2. **Style transfer** — rewrite normal text into genz slang.

## Approach

- Base model: [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Apache 2.0), fine-tuned with QLoRA (4-bit).
- Training data: synthetic examples (persona chat, style-transfer pairs, roleplay, slang Q&A) mixed with real slang datasets from the Hugging Face Hub for vocabulary grounding.
- Trained locally on an 8GB VRAM GPU.

See `configs/` for training hyperparameters and `model_card.md` for the published model documentation.

## Repo layout

```
configs/            LoRA + training config
data/
  raw/               downloaded HF grounding datasets (gitignored)
  synthetic/          authored training examples (per category, JSONL)
  processed/           merged/deduped train.jsonl + val.jsonl
scripts/             data build, training, eval, upload scripts
outputs/             checkpoints / adapter / merged model (gitignored)
model_card.md        Hugging Face model card
```

## Pipeline

```bash
python scripts/download_hf_datasets.py     # pull grounding datasets
python scripts/build_dataset.py            # merge, dedup, split synthetic data
python scripts/train_qlora.py --config configs/train_config.yaml
python scripts/eval_compare.py             # base vs fine-tuned qualitative comparison
python scripts/merge_lora.py               # merge LoRA into base weights
python scripts/push_to_hub.py              # upload adapter + merged repos
```
