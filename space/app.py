"""Gradio demo for abhisekm/genz-slang-model.

Two modes in one chat interface: persona chat and slang translation.
Runs on free CPU Space hardware, so responses are float32 and capped at
150 new tokens to keep latency reasonable.
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "abhisekm/genz-slang-model"

print(f"Loading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, device_map="cpu")
model.eval()

SYSTEM_PROMPTS = {
    "Persona chat": "You are a chill genz best friend who texts in genz slang.",
    "Slang translator": "Rewrite the user's text in genz slang. Keep the meaning the same.",
}

MAX_NEW_TOKENS = 150


def respond(message: str, history: list[dict], mode: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPTS[mode]}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
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
    return tokenizer.decode(output[0][input_ids.shape[-1] :], skip_special_tokens=True).strip()


demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Radio(list(SYSTEM_PROMPTS.keys()), value="Persona chat", label="Mode"),
    ],
    title="genz-slang-model",
    description=(
        "A [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) QLoRA fine-tune "
        "that chats in genz slang, or translates plain text into it — pick a mode below. "
        "Running on free CPU hardware, so replies take roughly 10-30 seconds. "
        "[Model card](https://huggingface.co/abhisekm/genz-slang-model) · "
        "[training write-up](https://github.com/AbhisekMishra/genz-model/blob/master/LEARNINGS.md)"
    ),
    examples=[
        ["yo whats good, im so bored today", "Persona chat"],
        ["what does rizz mean", "Persona chat"],
        ["I am extremely tired and want to go to sleep.", "Slang translator"],
        ["That was a great presentation, I'm impressed.", "Slang translator"],
    ],
)

if __name__ == "__main__":
    demo.launch()
