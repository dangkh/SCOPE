from math import nan
import pandas as pd
from tqdm import tqdm
import yaml
import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import random
from unsloth import FastLanguageModel
import torch
import json
from helper import build_item_item_knn, get_itemDesc, getUser_Interaction
from unsloth.chat_templates import get_chat_template



def get_message(system_prompt, content): 
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user",   "content": [{"type": "text", "text": content}]}
    ]
    return messages


def generate_summary(model, tokenizer, batchInfo):
    # ── Step 1: render each conversation to a string ──────────────────────
    allMessages = []
    for messages in batchInfo:
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        allMessages.append(input_text)

    # ── Step 2: tokenize with the inner text tokenizer ────────────────────
    tokenizer.tokenizer.padding_side = "left"
    if tokenizer.tokenizer.pad_token is None:
        tokenizer.tokenizer.pad_token = tokenizer.tokenizer.eos_token

    inputs = tokenizer.tokenizer(
        allMessages,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to("cuda")

    # ── Step 3: generate ──────────────────────────────────────────────────
    output = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.5, top_p=0.95, top_k=20,
        do_sample=False,   # ← must be True when using temperature/top_p/top_k
        pad_token_id=tokenizer.tokenizer.pad_token_id,
    )

    # ── Step 4: trim prompt tokens, decode only new tokens ────────────────
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output)
    ]
    output_texts = tokenizer.tokenizer.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_texts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='book', help='name of datasets')
    parser.add_argument('--LLM', type=str, default='gema', help='name of LLM to use: Llama or Gemma, Qwen')
    # output file name
    parser.add_argument('--output', '-o', type=str, default='user_profiles.json', help='output file name for user profiles')
    args, _ = parser.parse_known_args()
    print(args)

    # =========================
    # Load meta data
    # =========================
    meta_data = []
    file_path = f'./data/{args.dataset}/fullMeta_{args.dataset}.csv'
    metaDF = pd.read_csv(file_path)
    metaDF = pd.DataFrame(metaDF)
    unique_meta_asin = set(metaDF['asin'])
    print(f"[Meta] Unique ASINs: {len(unique_meta_asin)}")

    file_path = f'./data/{args.dataset}/{args.dataset}.inter'
    interDF = pd.read_csv(file_path, sep="\t", usecols=['userID', 'itemID', 'x_label'])
    interDF['userID'] = interDF['userID'].astype(int)
    interDF['itemID'] = interDF['itemID'].astype(int)

    # =========================
    # Preparing for users
    # =========================

    user_interactions = getUser_Interaction(interDF)
    # from user_interaction, get item degree
    item_degree = {}
    for u, items in user_interactions.items():
        for item in items:
            if item not in item_degree:
                item_degree[item] = 0
            item_degree[item] += 1
    # sort item theo degree giảm dần
    sorted_items = sorted(
        item_degree.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # số lượng top item
    high_ratio = 0.10
    num_high = int(len(sorted_items) * high_ratio)

    # lấy top 10%
    popular_items = set([
        item for item, degree in sorted_items[:num_high]
    ])
    # =========================
    # Profiling for user
    # =========================
    with open("src/prompts.yaml", "r") as f:
        all_prompts = yaml.safe_load(f)
    sys_prompt = all_prompts[args.dataset]['user']


    itemDesc = get_itemDesc(metaDF)

    fourbit_models = [
        "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit", # Qwen 14B 2x faster
        "unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit",
        "unsloth/Qwen3-8B-unsloth-bnb-4bit",
        "unsloth/Qwen3-14B-unsloth-bnb-4bit",
        "unsloth/Qwen3-32B-unsloth-bnb-4bit",

        # 4bit dynamic quants for superior accuracy and low memory use
        "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
        "unsloth/Phi-4",
        "unsloth/Llama-3.1-8B",
        "unsloth/Llama-3.2-3B",
        "unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit" # [NEW] We support TTS models!
    ] # More models at https://huggingface.co/unsloth

    selected_model = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"

    print(selected_model)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = selected_model,
        max_seq_length = 4096, # Choose any for long context!
        load_in_4bit = True,  # 4 bit quantization to reduce memory
        load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning = False, # [NEW!] We have full finetuning now!
        device_map = "balanced",
        # token = "hf_...", # use one if using gated models
    )

    tokenizer.tokenizer.padding_side = "left"
    if tokenizer.tokenizer.pad_token is None:
        tokenizer.tokenizer.pad_token = tokenizer.tokenizer.eos_token
    FastLanguageModel.for_inference(model)
    user_profiles = {}
    checkarray = []
    listUser = list(user_interactions.keys())
    users = listUser

    # your output filename for user profiles
    user_profile_path = f'./data/{args.dataset}/{args.output}'
    if os.path.exists(user_profile_path):
        with open(user_profile_path, 'r', encoding='utf-8') as f:
            user_profiles = json.load(f)
        print(f"Loaded existing user profiles from {user_profile_path}, current size: {len(user_profiles)}")
    q_message = []
    q_id = []
    batch_size = 8
    batch_messages = []
    for uid in tqdm(users):
        if str(uid) in user_profiles:
            continue
        u_is = user_interactions[uid]
        # u_items contain only items not appearing in popular_items
        u_items = [item for item in u_is if item not in popular_items]
        random.shuffle(u_items)
        itemInfo = "The user has purchased: \n"
        for item in u_items:
            itemInfo += itemDesc[item]

        messages = get_message(sys_prompt, itemInfo)
        q_message.append(messages)
        q_id.append(str(uid))
        if len(q_message) >= batch_size:
            batch_messages.append([q_id, q_message])
            q_message = []
            q_id = []
    

    if len(q_message) > 0:
        batch_messages.append([q_id, q_message])
        q_message = []
        q_id = []
        
    for batchId, batchInfo in tqdm(batch_messages):
        summary = generate_summary(model, tokenizer, batchInfo)
        for i, uid in enumerate(batchId):
            user_profiles[str(uid)] = { "summary": summary[i] }

        if (len(user_profiles)) % (batch_size * 10) == 0:
            with open(user_profile_path, 'w', encoding='utf-8') as f:
                json.dump(user_profiles, f, ensure_ascii=False, indent=4)

    with open(user_profile_path, 'w', encoding='utf-8') as f:
        json.dump(user_profiles, f, ensure_ascii=False, indent=4)
    
    # stat for candidate
    # print(np.mean(checkarray), np.min(checkarray), np.max(checkarray))
    


