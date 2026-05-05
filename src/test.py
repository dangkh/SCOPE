from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

tokenizer.tokenizer.padding_side = "left"
if tokenizer.tokenizer.pad_token is None:
    tokenizer.tokenizer.pad_token = tokenizer.tokenizer.eos_token

messages_batch = [
    # 1. Simple Q&A with a system prompt
    [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful geography expert."}]},
        {"role": "user",   "content": [{"type": "text", "text": "What is the capital of France?"}]},
    ],
    # 2. Multi-turn conversation with a different system prompt
    [
        {"role": "system",    "content": [{"type": "text", "text": "You are a concise scientific explainer. Keep answers under 3 sentences."}]},
        {"role": "user",      "content": [{"type": "text", "text": "What is quantum entanglement?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Quantum entanglement is a phenomenon where two particles become correlated."}]},
        {"role": "user",      "content": [{"type": "text", "text": "Can it be used for faster-than-light communication?"}]},
    ],
    # 3. No system prompt at all
    [
        {"role": "user", "content": [{"type": "text", "text": "Write a haiku about autumn leaves."}]},
    ],
    # 4. Long system prompt + detailed user request
    [
        {"role": "system", "content": [{"type": "text", "text": (
            "You are a senior software engineer specialising in Python. "
            "Always provide type hints, docstrings, and brief inline comments. "
            "Prefer readability over cleverness."
        )}]},
        {"role": "user", "content": [{"type": "text", "text": "Write a function that flattens a nested list."}]},
    ],
]

# ── Render text templates first, then tokenize manually ───────────────────
texts = [
    tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    for msgs in messages_batch
]

# ── Tokenize with the inner text tokenizer directly ────────────────────────
inputs = tokenizer.tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=2048,
).to("cuda")

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.tokenizer.pad_token_id,
    )

input_len = inputs["input_ids"].shape[1]
responses = tokenizer.tokenizer.batch_decode(
    output_ids[:, input_len:],
    skip_special_tokens=True,
)

for msgs, resp in zip(messages_batch, responses):
    print(f"User : {msgs[0]['content'][0]['text']}")
    print(f"Model: {resp}\n")