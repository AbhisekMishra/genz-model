---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
  - genz
  - slang
  - lora
  - qlora
  - peft
language:
  - en
---

# genz-slang-model

A [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) fine-tune (QLoRA) that talks in genz internet slang. Two modes, selected via system prompt:

1. **Persona chat** — free-form conversation in a genz voice.
2. **Style transfer** — rewrite normal text into genz slang.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "abhisekm/genz-slang-model"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

messages = [
    {"role": "system", "content": "You are a chill genz best friend who texts in genz slang."},
    {"role": "user", "content": "yo what should i do this weekend im so bored"},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
output = model.generate(inputs, max_new_tokens=200, do_sample=True, temperature=0.8, top_p=0.9)
print(tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True))
```

For style transfer, swap the system prompt to: `"Rewrite the user's text in genz slang. Keep the meaning the same."`

A LoRA-adapter-only version of this model (apply on top of the base model yourself) is available at `abhisekm/genz-slang-model-lora`.

## Training data

A mix of:
- Synthetic examples covering casual persona chat, style-transfer pairs, roleplay/reactions, slang Q&A, and mixed-register edge cases.
- Real slang term/definition/example data from Hugging Face Hub datasets (`MLBtrio/genz-slang-dataset` and related) used to ground vocabulary in authentic usage, and a capped, deduped, programmatically-converted sample from `Programmer-RD-AI/genz-slang-pairs-1k` and `thesherrycode/gen-z-slangs-translation` for style-transfer phrasing patterns.

## Training procedure

QLoRA (4-bit NF4), LoRA rank 16 / alpha 32 on all attention and MLP projections, 3 epochs, batch size 4 with gradient accumulation 8, learning rate 2e-4 (cosine schedule). Trained locally on a single 8GB VRAM GPU.

## Limitations

- Slang goes stale fast — this model reflects the vocabulary present in its training data at the time it was built, not real-time internet trends.
- Not intended for factual reliability, safety-critical, or professional use — it's a hobby/persona project.
- May inconsistently apply slang or occasionally revert to a more neutral register.
- Slang Q&A accuracy varies by term. Common/central terms (rizz, no cap, mid, bussin, npc, sigma, canon event, booked and busy, and dozens more) were specifically verified and reinforced during training. Less common or ambiguous terms (ones that overlap with unrelated, more common word meanings) may still get inaccurate definitions — this is an inherent long-tail limitation of a small (1.5B parameter) model fine-tuned on a few thousand examples, not something that can be fully eliminated without much larger scale training data.
- A few individual terms have proven unstable across retrains even after targeted reinforcement — notably **"sending me"** (should mean "made me laugh hard"), which has flipped between correct and incorrect across different training runs on the same dataset. Treat single-term slang definitions as informative but not authoritative.
- Inherits any limitations/biases of the base model, `Qwen/Qwen2.5-1.5B-Instruct`.

## License

Apache 2.0, inherited from the base model.
