# Project Log — Fine-Tuning a GenZ Slang Model

A complete, chronological record of building [`abhisekm/genz-slang-model`](https://huggingface.co/abhisekm/genz-slang-model) — a QLoRA fine-tune of `Qwen2.5-1.5B-Instruct` trained locally on an 8GB laptop GPU. Written to be read start to finish as a learning resource: every decision, every bug, every number, in the order it actually happened. For a condensed interview-prep version of the same story, see the [study guide artifact](https://claude.ai/code/artifact/fc3e2f74-ea4f-4b0c-aee5-b6a75769164c).

---

## 0. The brief

Build a small open LLM that:
1. Chats in a genz-slang persona (free-form conversation).
2. Rewrites plain text into genz slang (style transfer).

Both behaviors in **one model**, switched by system prompt. No LLM API budget for data generation. Publish to Hugging Face.

---

## 1. Planning

Before writing any code, a Plan agent was used to research and settle the open technical questions:

- **Environment**: WSL2 was the tentative default recommendation for QLoRA on Windows (historically better `bitsandbytes`/kernel support on Linux) — with instructions to smoke-test native Windows first and fall back only if it broke. In practice, native Windows CUDA worked fine (see §3), so WSL2 was never needed.
- **Base model shortlist**: Qwen2.5-1.5B-Instruct (Apache 2.0, ungated), Llama-3.2-3B-Instruct (gated, Llama license), SmolLM2-1.7B-Instruct (Apache 2.0). Qwen2.5-1.5B-Instruct was chosen — see §2.
- **Data plan**: synthetic examples authored directly (no API budget) + real HF slang datasets for grounding.

---

## 2. Key decisions and why

| Decision | Reasoning |
|---|---|
| **Qwen2.5-1.5B-Instruct** as base | Apache 2.0 license (the 3B sibling in the same family is under a *more restrictive* non-Apache license — avoided deliberately), fully ungated (no manual access request, unlike Llama), and small enough to fit 4-bit-quantized in 8GB VRAM with headroom. |
| **QLoRA**, not full fine-tuning | Full fine-tuning needs gradients + Adam optimizer state (2 extra values per parameter) + activations for every one of 1.5B parameters — 20–30GB+ easily. QLoRA freezes the base model, quantizes it to 4-bit, and trains only small injected LoRA adapter matrices. |
| **Native Windows**, not WSL2 | `nvidia-smi` confirmed the GPU (RTX 4060 Laptop, 8188 MiB, driver 566.26), and `pip install torch --index-url .../cu124` gave a working CUDA build directly. No need to add WSL2 complexity once the smoke test passed. |
| **Synthetic + grounding data**, not scraping | No LLM API budget ruled out runtime generation at scale. Authored a pilot batch by hand for tone approval, then parallelized authoring across independent subagents with a fixed schema. |

---

## 3. Environment setup

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verified: `torch 2.6.0+cu124`, `cuda available: True`, `NVIDIA GeForce RTX 4060 Laptop GPU`.

Installed the rest of the stack (`transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`, `datasets`, `huggingface_hub`, `pyyaml`, `numpy`). What actually installed was **much newer** than the plan assumed:

| Library | Planned (floor) | Actually installed |
|---|---|---|
| `transformers` | ≥4.46 | **5.15.0** |
| `trl` | ≥0.12 | **1.10.0** |
| `peft` | ≥0.13 | 0.20.0 |
| `bitsandbytes` | ≥0.44 | 0.50.1 |

This version gap directly caused three of the bugs in §6.

---

## 4. Repo scaffolding

Created the project structure *before* running anything: `configs/`, `data/{raw,synthetic,processed}/`, `scripts/`, `outputs/{checkpoints,adapter,merged_model}/`, `.gitignore`, `README.md`, `requirements.txt`.

Wrote all pipeline scripts up front, each with a single clear job:

| Script | Job |
|---|---|
| `download_hf_datasets.py` | pull grounding datasets from the HF Hub |
| `build_dataset.py` | merge all `data/synthetic/*.jsonl`, dedupe, stratified train/val split |
| `train_qlora.py` | QLoRA fine-tune, `--max_steps` flag for short pilot runs |
| `merge_lora.py` | fold the trained adapter into the base weights |
| `eval_compare.py` | base-vs-fine-tuned generation on a fixed held-out prompt set |
| `push_to_hub.py` | upload both the adapter-only and merged repos |

---

## 5. Data schema design

Every training example: `{"category": "...", "messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`.

**Why `messages`**: this is the schema `tokenizer.apply_chat_template()` and TRL's `SFTTrainer` expect natively — the same shape most HF chat-fine-tuning datasets use. Reusing it meant: no custom prompt-formatting code, correct assistant-only loss masking (only `assistant` turns are trained on, `system`/`user` text isn't), and the two capabilities (persona chat vs. translation) could live entirely in the choice of `system` message rather than needing two different schemas.

**Why `category`**: pure pipeline bookkeeping, never seen by the model or the trainer. Used only inside `build_dataset.py` for (a) stratified train/val splitting so validation isn't skewed toward whichever category has the most examples, (b) per-category stats after every merge, and (c) giving parallel authoring subagents an unambiguous label to stamp on their own output.

Five categories, chosen so the model would learn *when to apply slang and when not to*:
`casual_chat`, `slang_translation`, `roleplay_reactions`, `slang_qna`, `edge_cases` (formal/factual/crisis-adjacent prompts, deliberately included so the model doesn't slang-ify everything).

---

## 6. Pilot data batch — Checkpoint 1

Hand-authored **118 examples** across all five categories (36 casual_chat, 32 slang_translation, 19 roleplay_reactions, 19 slang_qna, 12 edge_cases) *before* investing in a large-scale generation pass. Presented a sample to the user for tone approval before scaling up — this is the point where a wrong style choice would have been cheapest to catch and most expensive to discover after generating thousands of examples.

---

## 7. Scaling the dataset — 6 parallel subagents

Once the pilot was approved, launched **6 subagents in parallel**, each authoring one category/topic slice with a fully self-contained prompt (schema, exact system-prompt strings, a hand-typed vocabulary list of ~70 real slang terms, style calibration examples, hard rules against vulgar/toxic terms, anti-repetition instructions, and exact output file paths):

| Agent | Category | Target | Actual |
|---|---|---|---|
| 1 | casual_chat (topics: school/family/health/decisions) | 600 | 600 |
| 2 | casual_chat (topics: gaming/dating/money/career) | 600 | 600 |
| 3 | slang_translation (topics A) | 600 | 600 |
| 4 | slang_translation (topics B) | 600 | 600 |
| 5 | roleplay_reactions | 800 | 800 |
| 6 | slang_qna (500) + edge_cases (300) | 800 | 800 |

**Total: 4,118 lines across 30 files** — matched the target exactly.

In parallel, downloaded 4 real HF slang datasets into `data/raw/` for "grounding": `MLBtrio/genz-slang-dataset` (1,779 rows), `Programmer-RD-AI/genz-slang-pairs-1k` (1,005 rows), `thesherrycode/gen-z-slangs-translation` (140 rows), `nikesh66/Slang-Dataset` (5,000 rows, tweet/is-slang classification — not directly usable for generation or definitions).

> **Honesty note, found later (§20):** these downloaded datasets were previewed for schema but never actually wired into the generation pipeline at this point. The vocabulary list handed to the subagents was typed from memory, not extracted from the files. `build_dataset.py` never reads `data/raw/` at all. This was a real gap between the stated plan and the implementation — fixed in §21.

---

## 8. Dataset build (merge → dedupe → split)

`build_dataset.py`: loads every `data/synthetic/*.jsonl`, hashes normalized text for exact-duplicate removal, splits 90/10 **stratified by category** (so validation isn't dominated by whichever category is largest), prints per-category counts and character-length percentiles.

First run: **4,118 loaded → 2 exact duplicates removed → 4,116 kept → 3,707 train / 409 val.**

---

## 9. Training configuration

QLoRA via `transformers` + `peft` + TRL's `SFTTrainer` (not Unsloth — the plain HF stack was used directly).

| Hyperparameter | Value |
|---|---|
| Quantization | 4-bit NF4, double quantization, bf16 compute dtype |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | all attention + MLP projections |
| Max sequence length | 1024 |
| Batch size / grad. accumulation | 4 / 8 (effective batch 32) |
| Learning rate / schedule | 2e-4, cosine |
| Warmup | 30 steps |
| Epochs | 3 |
| Optimizer | paged AdamW 8-bit |
| Gradient checkpointing | on |
| Seed | 42 |

Trainable parameters: **18,464,768 of 1,562,179,072 total — 1.18%.**

---

## 10. Pilot training run — and three library-drift bugs

Ran a 20-step pilot (`--max_steps 20`) to validate the pipeline before committing to a full run. It failed twice before succeeding, both times due to the newer-than-expected library versions from §3:

**Bug 1 — `SFTConfig.__init__() got an unexpected keyword argument 'warmup_ratio'`**
Caught *before* running, by inspecting the installed `SFTConfig` signature directly (`inspect.signature`) rather than trusting memorized docs. `warmup_ratio` doesn't exist in this TRL version; only `warmup_steps` does. Fixed: config changed to `warmup_steps: 30`, script updated to pass it through.

**Bug 2 — `TypeError: SFTConfig.__init__() got an unexpected keyword argument` on `max_seq_length`**
Same root cause — `max_seq_length` was renamed to `max_length` in this TRL version. Fixed by passing `max_length=cfg["max_seq_length"]` (kept the friendlier name in the YAML config, mapped it at the call site).

**Bug 3 — `AttributeError` / `KeyError: 'shape'` inside `model.generate()`**
`tokenizer.apply_chat_template(..., return_tensors="pt")` now returns a `BatchEncoding` dict, not a raw tensor, in this `transformers` version. Fixed by extracting `inputs["input_ids"]` before passing to `.generate()`.

After all three fixes, the pilot run succeeded: loss `4.444 → 2.845` over 20 steps, eval_loss `1.967`, eval_mean_token_accuracy `0.6087`, no NaN/divergence, ~3.4s/step (~20 min projected for a full 3-epoch run). Cleared for the full run.

---

## 11. First full training run

3 epochs, ~1,247s (~20.8 min) wall clock.

| Epoch | Eval loss | Note |
|---|---|---|
| 0.86 | 1.299 | |
| 1.29 | 1.229 | |
| 1.73 | 1.184 | lowest point |
| 2.16 | 1.194 | starts flattening/rising slightly |
| 2.59 | 1.186 | |
| 3.00 | 1.187 | |

Train loss kept falling to ~0.85 while eval loss plateaued around 1.18–1.19 after epoch ~1.7 — a classic (and here, mild, not alarming) overfitting signal: the model keeps fitting the training set slightly further without generalizing further. Final `train_loss: 1.249`.

---

## 12. First evaluation — and an encoding bug

Built a 25-prompt held-out set (`data/eval_prompts.json`) spanning persona chat, translation, slang Q&A, and **plain factual questions** (capital of France, photosynthesis, 15% of 240, resume tips, Romeo & Juliet) specifically to catch catastrophic forgetting.

First run crashed: `UnicodeEncodeError: 'charmap' codec can't encode characters` — Windows console defaults to `cp1252`, and the model occasionally generated a character outside that codepage. Fixed with `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` at the top of `eval_compare.py`.

**Results after the fix:**
- Style transfer: solid (`"I am extremely tired..." → "im straight up cooked and wanna go to sleep rn"`).
- Persona chat: warmer, more concise, genuinely useful advice, correct tone.
- Slang Q&A: **two flat-out wrong definitions.** "rizz" came back as *"the opposite of quiet... loud and boisterous"* (should be charisma). "no cap" came back as *"not stressing about the situation, just saying no honestly"* (should be "not lying, for real").
- No catastrophic forgetting: all five factual prompts stayed accurate and coherent.

---

## 13. Root-causing the wrong definitions

Grepped the actual training data for the exact terms rather than guessing:

```
grep -il "rizz" data/synthetic/*.jsonl   → 2 matches in ~4,118 examples
grep -il "no cap" (as a definition, not just usage) → 1 match
```

**The training data itself had the correct definitions.** The bug wasn't bad data — it was **insufficient repetition**. LoRA only updates 1.2% of the model's parameters (§9); a fact appearing once or twice among ~4,000 examples over 3 epochs (3-6 gradient touches) isn't enough signal to overwrite a base model's existing, apparently-wrong prior belief about that term. This became the central lesson of the whole project: **fine-tuning nudges a distribution, it doesn't overwrite a fact unless that fact shows up repeatedly, in varied phrasing.**

---

## 14. Reinforcement round 1

Hand-authored **73 new `slang_qna` examples**, concentrated on the wrong terms with *varied phrasing* (not the same sentence repeated): rizz ×8, no cap/cap ×6, mid ×5, plus 2-3 each for ~20 other terms (bussin, delulu, npc, sus, based, cringe, ick, glow up, ate, understood the assignment, main character energy, hits different, rent free, W/L, lowkey/highkey, say less, touch grass, brainrot, aura, cooked, goated, ratio).

Rebuilt dataset (4,191 → 4,189 after dedup; slang_qna grew 468 → 533), retrained (same ~20 min shape, eval_loss plateau unchanged), re-evaluated: rizz and mid now correct; "no cap" correct on its core claim but with one fabricated add-on (*"opposite of lowkey"* — not true).

**First publish**: merged the adapter (`peft.merge_and_unload()`), uploaded both `abhisekm/genz-slang-model` (merged, standalone) and `abhisekm/genz-slang-model-lora` (adapter-only) to Hugging Face. Verified end-to-end by downloading the model *fresh from the Hub* (not local cache) and confirming a correct "rizz" generation.

---

## 15. User-requested spot check → 11 more prompts → 3 new misses

Asked to "try a few more prompts." Found:
- *"someone said i ate that outfit"* → model said it meant the outfit was **boring** — backwards. "Ate" is a strong compliment.
- *"That was a terrible decision"* translated using **"ick"** — wrong context; "ick" is specifically a romantic turn-off, not general disapproval.
- *"whats a soft launch"* → vague/generic answer; term simply wasn't in training data at all.

---

## 16. Broad stress test — 65 prompts, ~50 new terms

Asked to test "many examples covering all possible aspects." Built a 72-prompt stress test (`stress_test.py`) covering: 5 spot-check terms, 34 brand-new terms never touched in training, 15 translation prompts specifically designed to test whether the model reaches for the *right* slang term in context (not just define it correctly in isolation), 10 persona prompts embedding slang naturally, 5 edge cases.

**Run with sampling (`temperature=0.7`).** Results: ~13 clearly wrong definitions (sigma, beige flag, yeet, simp, periodt, boy math, canon event, booked-and-busy, "sending me," pmo, fit check, chef's kiss, and stan — correct core but with a fabricated extra word, *"arstrot"*, tacked on). Also found **application** bugs: "ick" misused twice more for non-romantic embarrassment instead of "cringe"; "left on read," "rent free," and "ratio" were correctly *defined* elsewhere but not *reached for* in translation even in textbook-matching sentences; "locking in" got used backwards (describing doing *nothing* productive).

**What was already fine**: green/red flag, bet, soft launch (once explained), ghosted, left-on-read (as a Q&A definition), drip, big flex, receipts, spill the tea, doom scrolling, lwk, ong, bffr, icl, gaslighting, and all edge-case handling.

---

## 17. Reinforcement round 2 — and a transient CUDA crash

Wrote 48 new examples (`slang_qna_reinforced_2.jsonl`): 35 Q&A examples across sigma/beige-flag/yeet/simp/periodt/boy-math/canon-event/booked-and-busy/sending-me/pmo/fit-check/chefs-kiss/stan (~3 each), plus 4 more "ate"-in-outfit-context examples, plus **13 translation-category contrastive examples** — pairs that correctly use "ick" for romantic turn-offs *and* correctly use "cringe" instead for general embarrassment, examples anchoring "left on read"/"rent free"/"ratio" into the translation task itself (not just Q&A), and examples using "locked in" correctly (focus, not idleness).

Rebuilt (4,239 → 4,237 after dedup; slang_qna → 565). First retraining attempt **crashed at step 228/360 (63%)** with `RuntimeError: CUDA error: unknown error` — a driver-level transient fault, not a code bug (the machine had gone idle/slept overnight between sessions; date rolled Aug 16 → Aug 17 mid-conversation). Confirmed the GPU was healthy afterward (`nvidia-smi`: 36°C, 2% util, no stuck processes) and simply retried — succeeded cleanly, 1,322s.

---

## 18. The sampling-vs-greedy-decoding detour (the most important methodology lesson)

Re-ran the stress test against the new checkpoint (still with `temperature=0.7` sampling) and it looked like several **previously-fixed** terms had regressed — "ong" flipped from correct to wrong, for example. Before chasing what looked like a new bug, re-tested the same suspect terms with **greedy decoding** (`do_sample=False` — always the single highest-probability token, fully deterministic) instead of sampling.

Most of the "regressions" disappeared. `stan` came back clean (no more fabricated word), `locked in` and `icl`/`lwk` were correct, `beige flag` and `chef's kiss` were reasonable. **Sampling-based generation is inherently noisy** for evaluating factual/definitional correctness — it randomly draws from the model's output distribution rather than showing its single most-confident answer. An entire round of apparent regressions was largely sampling variance, not a real change in what the model had learned.

Genuinely still wrong under greedy decoding (the real, non-noise signal): **sigma, canon event, booked-and-busy, "sending me,"** plus one real regression: **"ong"** (was correct before, now wrong).

---

## 19. Reinforcement round 3 — light touch, partial results

22 targeted examples (`slang_qna_reinforced_3.jsonl`, ~4 each) for sigma, canon event, booked-and-busy, "sending me," ong, hbu. Rebuilt (4,261 → 4,259; slang_qna → 585), retrained (1,333s), re-verified with greedy decoding.

**Fixed**: ong, locked-in (still stable), bffr (still stable), beige flag (improved to acceptable).
**Still wrong, despite direct reinforcement**: sigma, canon event, booked-and-busy, "sending me." A light touch (~4 examples) wasn't enough for these four specifically — they each collide with a much more common existing meaning the base model already knows strongly (sigma = math symbol, canon = fiction terminology, "booked" = calendar term, "sent" = delivery/shipping), so a handful of counter-examples couldn't outweigh that competing signal.

---

## 20. Reinforcement round 4 — the heavier push

Given the choice between stopping here (documenting a limitation), a heavier push, or excluding the terms entirely, the heavier-push option was chosen: **46 examples**, ~15 each for sigma/canon-event/booked-and-busy/"sending me," with **explicit contrastive corrections** baked into the answers ("it's not an insult," "not about a TV show," "one phrase, not two meanings," "nothing to do with delivery or timing").

Rebuilt (4,307 → 4,305; slang_qna → 626), retrained (1,306s), then verified two ways:
1. **Exact phrasings seen in training** — all four correct.
2. **Novel phrasings never seen in training** ("my coworker has such sigma vibes," "was getting dumped a canon event for you," "that tiktok sent me so hard") — all four still correct, confirming the model had learned the *concept*, not just memorized specific sentences.

A final broad regression check confirmed rizz/no cap/mid/yeet/boy-math/bffr/hbu all held or improved, with `chef's kiss`, `glow down`, and `beige flag` left as known, minor, unresolved long-tail imperfections — deliberately not chased further given diminishing returns after 4 reinforcement rounds.

---

## 21. Closing the honesty gap: actually using the raw datasets

A direct question — *"did you refer to the raw datasets while generating synthetic data?"* — prompted a `grep` check that confirmed §7's honesty note: `data/raw/` was downloaded and previewed, but never read by any generation step. Two follow-up questions shaped the fix:

**"Why `messages`/`category`?"** — answered in §5 above (this was asked and answered mid-project, included here for completeness).

**"How do we use the raw datasets without burning a lot of tokens?"** — the answer: never pipe raw rows through an LLM context. Two treatments depending on dataset shape:
1. **Already-paired datasets** (`Programmer-RD-AI`'s `normal`/`gen_z` columns, `thesherrycode`'s `Plain English`/`Gen-Z Slang` columns) → convert **programmatically** straight into the training schema. Zero LLM tokens — pure Python.
2. **Definitional datasets** (`MLBtrio`'s term/description/example rows) → compress locally (dedupe, truncate) into a small reference file, and *query it* for specific terms via `grep` rather than reading it in full.

**Implementation** — two new scripts:

`scripts/convert_pair_datasets.py`: loads each pair-dataset, filters (length bounds 8-220 chars, banned-substring list, dedup by normalized source text), caps `Programmer-RD-AI` at 200 examples (it was itself GPT-4.1-Nano-generated, so deliberately kept as a *minority* share rather than let one other model's phrasing dominate), writes directly to `data/synthetic/slang_translation_from_raw.jsonl` in the `messages`/`category` schema.

```
[ok] Programmer-RD-AI__genz-slang-pairs-1k.jsonl: 1005 passed filters -> 200 kept (cap=200)
[ok] thesherrycode__gen-z-slangs-translation.jsonl: 137 passed filters -> 88 kept (cap=None)
Wrote 288 examples -> data/synthetic/slang_translation_from_raw.jsonl
```

`scripts/build_vocab_reference.py`: reads `MLBtrio`, dedupes by lowercased term (first occurrence wins), truncates descriptions to 140 characters, writes a sorted markdown table.

```
1779 raw rows -> 1571 unique terms -> data/vocab_reference.md   (53KB — larger than
                                        expected; the source data has only ~12% duplication)
```

Spot-checked the new reference file with `grep` for the terms that had needed heavy manual reinforcement: **"sigma," "no cap," and "canon event" are not in this dataset at all.** "Rizz" is, with a reasonable definition. This explained, after the fact, *why* those specific terms needed so much hand-authored reinforcement — no amount of using this particular dataset would have covered them; they were genuinely outside its scope.

---

## 22. Merging the raw-derived data and retraining

Rebuilt the dataset with the new 288 examples folded in (slang_translation grew 1,121 → 1,380 in train): **4,595 loaded → 4,593 after dedup → 4,136 train / 457 val.** Retrained (1,408s, `train_loss: 1.204` — slightly lower than prior rounds).

Greedy-decode verification: the new raw-derived translations were natural and fluent; rizz/sigma/canon-event/booked-and-busy all held stable from §20. But **"sending me" regressed again** — back to a vague, wrong answer, despite being confirmed fixed (and generalizing) in §20's verification. This was under greedy decoding, so it wasn't sampling noise — it's a genuine instability: retraining on a larger/reshuffled dataset (even with the same fixed seed) can shift how a fragile term's few reinforcing examples get batched relative to competing signal elsewhere in a growing dataset.

**Decision**: rather than chase it again (this was the 6th full training run), documented it explicitly as a known-flaky term in the model card and shipped:

> "A few individual terms have proven unstable across retrains even after targeted reinforcement — notably 'sending me' ... which has flipped between correct and incorrect across different training runs on the same dataset. Treat single-term slang definitions as informative but not authoritative."

Re-merged, re-uploaded to both Hugging Face repos.

---

## 23. Publishing to GitHub

Confirmed `git` and `gh` were available and already authenticated (`gh auth status` → logged in). Reviewed `.gitignore` before staging anything (`.venv/`, `data/raw/`, `data/processed/*.jsonl`, `outputs/checkpoints/`, `outputs/merged_model/`, adapter weight files — all excluded; only small config/text artifacts from the adapter kept). Confirmed via `git status` after `git add -A` that nothing large or secret was about to be committed (the HF write token was only ever passed as an environment variable to specific commands, never written to a file).

```
git init
git add -A
git commit -m "Fine-tune Qwen2.5-1.5B-Instruct into a genz-slang persona/translator model"
gh repo create genz-model --public --source=. --remote=origin --push
```

→ [github.com/AbhisekMishra/genz-model](https://github.com/AbhisekMishra/genz-model)

Later additions, each committed and pushed separately: a Hugging Face badge in the README, the raw-dataset conversion scripts and their outputs (§21-22), and documentation updates.

---

## 24. The study guide artifact

Produced a separate, curated interview-prep document (not this file) — a designed HTML page covering the same project through a narrower lens: the three defensible technical decisions, a LoRA/QLoRA explainer, the hyperparameter table, the four-round debugging story with a real SVG chart plotted from the actual §11 loss numbers, a glossary, and a rehearsed Q&A section. Published as a private Claude Artifact.

---

## Key lessons (condensed)

1. **A fact needs repetition to survive fine-tuning.** LoRA touches ~1% of parameters; one or two examples out of thousands isn't enough to overwrite what the base model already "believes" about a token sequence — especially when that belief collides with a much more common existing meaning (sigma, canon, booked, sent).
2. **Vary the phrasing, not just the count.** A term seen only in one exact sentence doesn't reliably generalize to a different phrasing of the same question. Always test with phrasings the training data never used.
3. **Sampling and greedy decoding measure different things.** Temperature-based sampling is right for creative generation; it is the wrong tool for evaluating factual/definitional correctness, because it introduces randomness that can look exactly like a regression. Greedy decoding gives the model's actual most-confident answer.
4. **A held-out eval set needs a forgetting check, not just a task check.** Plain factual prompts unrelated to the fine-tuning task are what catch catastrophic forgetting — and their absence would have hidden it.
5. **Verify library APIs against what's actually installed**, not against memorized docs or training data — `inspect.signature()` caught two breaking changes before they caused failures.
6. **Distinguish a code bug from a transient hardware fault.** The CUDA crash mid-training wasn't a logic error; `nvidia-smi` confirmed GPU health, and a plain retry was the correct fix — no code change needed.
7. **Don't let stated design intent silently diverge from implementation.** The raw datasets were "planned" as grounding material but never actually wired in for four rounds of work — caught only because someone asked directly. Worth periodically re-checking that a pipeline still does what its own documentation claims.
8. **Processing data locally (pure code) vs. reading it through an LLM context are very different costs.** Programmatic conversion of already-structured data is free in tokens; even a "compressed" reference table can still be tens of thousands of tokens if the source data doesn't dedupe well — measure, don't assume.
9. **Know when to stop.** Four reinforcement rounds fixed the large majority of found issues; a few long-tail terms (chef's kiss, glow down, beige flag, and the persistently flaky "sending me") were consciously left as documented limitations rather than chased indefinitely — a real engineering tradeoff between marginal quality and time spent.

---

## Full script inventory

| Script | Purpose |
|---|---|
| `download_hf_datasets.py` | Download the 4 grounding datasets from the HF Hub into `data/raw/` |
| `convert_pair_datasets.py` | Programmatically convert already-paired HF datasets into training examples (no LLM calls) |
| `build_vocab_reference.py` | Compress the MLBtrio dataset into a deduped, queryable term-reference file |
| `build_dataset.py` | Merge, dedupe, and stratified-split all synthetic data into train/val |
| `train_qlora.py` | QLoRA fine-tune (`--max_steps` for pilot runs) |
| `eval_compare.py` | Base-vs-fine-tuned generation over the fixed held-out prompt set |
| `merge_lora.py` | Merge the trained adapter into standalone base weights |
| `push_to_hub.py` | Upload both the adapter-only and merged repos to Hugging Face |

## Full data inventory

~4,600 examples across `data/synthetic/*.jsonl`: the hand-authored 118-example pilot batch, ~4,000 examples from 6 parallel subagents (§7), 4 rounds of hand-authored reinforcement targeting specific wrong definitions (§14, §17, §19, §20 — 73 + 48 + 22 + 46 examples), and 288 examples programmatically converted from real HF datasets (§21). Plus `data/vocab_reference.md`, a 1,571-term compressed reference table built from the MLBtrio dataset.
