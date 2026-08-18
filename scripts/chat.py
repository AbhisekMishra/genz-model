"""Interactive terminal chat with the published genz-slang model.

Usage:
    python scripts/chat.py                        # loads from the Hugging Face Hub
    python scripts/chat.py --local                # loads outputs/merged_model instead
    python scripts/chat.py --model some/other-id   # loads a different HF repo

Commands while chatting:
    /mode persona      switch to persona-chat system prompt (default)
    /mode translate     switch to slang-translation system prompt
    /reset              clear conversation history
    /quit               exit
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent

SYSTEM_PROMPTS = {
    "persona": "You are a chill genz best friend who texts in genz slang.",
    "translate": "Rewrite the user's text in genz slang. Keep the meaning the same.",
}

MAX_NEW_TOKENS = 200


def load_model(model_id: str):
    print(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
    return tokenizer, model


def generate(tokenizer, model, system: str, history: list[dict]) -> str:
    messages = [{"role": "system", "content": system}] + history
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
        model.device
    )
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="abhisekm/genz-slang-model", help="HF repo id to load")
    parser.add_argument("--local", action="store_true", help="load outputs/merged_model instead")
    args = parser.parse_args()

    model_id = str(ROOT / "outputs" / "merged_model") if args.local else args.model
    tokenizer, model = load_model(model_id)

    mode = "persona"
    history: list[dict] = []
    print(f"\nReady. Mode: {mode}. Type /quit to exit, /mode translate to switch, /reset to clear history.\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/reset":
            history = []
            print("(history cleared)")
            continue
        if user_input.startswith("/mode"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1] in SYSTEM_PROMPTS:
                mode = parts[1]
                history = []
                print(f"(mode -> {mode}, history cleared)")
            else:
                print(f"usage: /mode persona | /mode translate  (current: {mode})")
            continue

        history.append({"role": "user", "content": user_input})
        reply = generate(tokenizer, model, SYSTEM_PROMPTS[mode], history)
        history.append({"role": "assistant", "content": reply})
        print(f"model> {reply}\n")


if __name__ == "__main__":
    main()
