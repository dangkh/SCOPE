from math import nan
import pandas as pd
from tqdm import tqdm
import yaml
import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import random
import warnings
from unsloth import FastLanguageModel
import torch
import json
from helper import build_item_item_knn, get_itemDesc, getUser_Interaction
from unsloth.chat_templates import get_chat_template

# Suppress a known Unsloth/PyTorch output-resize deprecation warning.
warnings.filterwarnings(
    "ignore",
    message=r"An output with one or more elements was resized since it had shape .*",
    category=UserWarning,
    module=r"unsloth\.kernels\.utils",
)



def get_message(system_prompt, content, use_list_format=False): 
    if use_list_format:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user",   "content": [{"type": "text", "text": content}]}
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": content}
        ]
    return messages


def generate_summary(model, tokenizer, batchInfo, is_qwen3=False):
    # ── Step 1: render each conversation to a string ──────────────────────
    allMessages = []
    chat_template_kwargs = dict(tokenize=False, add_generation_prompt=True)
    if is_qwen3:
        chat_template_kwargs["enable_thinking"] = False
    for messages in batchInfo:
        input_text = tokenizer.apply_chat_template(
            messages,
            **chat_template_kwargs,
        )
        allMessages.append(input_text)

    # Use underlying tokenizer when present (e.g., unsloth wrappers),
    # otherwise use the tokenizer directly (e.g., Qwen2TokenizerFast).
    base_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    # ── Step 2: tokenize with the selected text tokenizer ─────────────────
    base_tokenizer.padding_side = "left"
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    inputs = base_tokenizer(
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
        pad_token_id=base_tokenizer.pad_token_id,
    )

    # ── Step 4: trim prompt tokens, decode only new tokens ────────────────
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output)
    ]
    output_texts = base_tokenizer.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_texts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='book', help='name of datasets')
    parser.add_argument('--LLM', type=str, default='4B', help='name of LLM to use: Llama or Gemma, Qwen')
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

    # for each item, find top-k similar items
    top_k = 10
    item_item_path = f'./data/{args.dataset}/item_top{top_k}item.npy'
    if os.path.exists(item_item_path):
        print(f"{item_item_path} exists, skip building item-item knn.")
        item_kitem = np.load(item_item_path)
    else:
        raise ValueError(f"{item_item_path} does not exist, please run preprocess.py to build it.")


    model_map = {
        "gema":  "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        "4B":    "unsloth/Qwen3-4B-Instruct-2507",
        "llama": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    }
    if args.LLM not in model_map:
        raise ValueError(f"Unknown LLM '{args.LLM}'. Choose from: {list(model_map.keys())}")
    selected_model = model_map[args.LLM]
    is_qwen3 = args.LLM == "4B"
    is_gemma = args.LLM == "gema"

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

    base_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    base_tokenizer.padding_side = "left"
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    FastLanguageModel.for_inference(model)
    user_profiles = {}
    checkarray = []
    listUser = list(user_interactions.keys())
    users = listUser

    user_profile_path = f'./data/{args.dataset}/batch_{args.LLM}_usr_prf.json'
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

        messages = get_message(sys_prompt, itemInfo, use_list_format=is_gemma)
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
        summary = generate_summary(model, tokenizer, batchInfo, is_qwen3=is_qwen3)
        for i, uid in enumerate(batchId):
            user_profiles[str(uid)] = { "summary": summary[i] }

        if (len(user_profiles)) % (batch_size * 10) == 0:
            with open(user_profile_path, 'w', encoding='utf-8') as f:
                json.dump(user_profiles, f, ensure_ascii=False, indent=4)
            # break # for debug, only run 1 batch

    with open(user_profile_path, 'w', encoding='utf-8') as f:
        json.dump(user_profiles, f, ensure_ascii=False, indent=4)
    
    # stat for candidate
    # print(np.mean(checkarray), np.min(checkarray), np.max(checkarray))
    


