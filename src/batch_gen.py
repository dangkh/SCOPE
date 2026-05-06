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
    parser.add_argument('--sample', type=bool, default=True, help='whether to sample or full generation')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for LLM inference')
    parser.add_argument('--local', type=bool, default=False, help='whether to use local or global prompt')
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

    # =========================
    # Profiling for user
    # =========================
    with open("src/prompts.yaml", "r") as f:
        all_prompts = yaml.safe_load(f)
    
    colName = None
    if args.local:
        colName = "profile"
    itemDesc = get_itemDesc(metaDF, colname=colName)

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
    users = list(user_interactions.keys())

    if args.local:
        sys_prompt = all_prompts[args.dataset]['local']
        promptName = "localPrompt"
    else:        
        sys_prompt = all_prompts[args.dataset]['global']
        promptName = "globalPrompt"
        # load similar user from user_top10user.npy
        top10user_path = f'./data/{args.dataset}/user_top10user.npy'
        if os.path.exists(top10user_path):
            top10user = np.load(top10user_path, allow_pickle=True)
    if args.sample:
        sampleName = "sample"
    else:
        sampleName = ""
    user_profile_path = f'./data/{args.dataset}/{sampleName}_{promptName}_{args.LLM}_usr_prf.json'
    if os.path.exists(user_profile_path):
        with open(user_profile_path, 'r', encoding='utf-8') as f:
            user_profiles = json.load(f)
        print(f"Loaded existing user profiles from {user_profile_path}, current size: {len(user_profiles)}")
    q_message = []
    q_id = []
    batch_size = args.batch_size
    batch_messages = []
    for uid in tqdm(users):
        if str(uid) in user_profiles:
            continue
        u_items = user_interactions[uid]
        random.shuffle(u_items)
        # local prompt with only user
        itemInfo = "The user has purchased: \n"
        for item in u_items[-10:]:
            itemInfo += itemDesc[item]
        itemInfo += "\n"
        # infomation for global prompt with similar users
        if not args.local:
            userID = users.index(uid)
            topk_users = top10user[userID]
            # add information from similar users
            aux_info = "Purchase history of similar users: \n"
            for simU in topk_users:
                user_items = user_interactions[simU]
                random.shuffle(user_items)
                itemInfo += f"Similar user {simU} has purchased: \n"
                for item in user_items:
                    itemInfo += itemDesc[item]
                itemInfo += "\n"
            aux_info += itemInfo
            itemInfo += aux_info


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
        if args.sample:
            break
    
        if (len(user_profiles)) % (batch_size * 10) == 0:
            with open(user_profile_path, 'w', encoding='utf-8') as f:
                json.dump(user_profiles, f, ensure_ascii=False, indent=4)
    with open(user_profile_path, 'w', encoding='utf-8') as f:
        json.dump(user_profiles, f, ensure_ascii=False, indent=4)
    
    # stat for candidate
    # print(np.mean(checkarray), np.min(checkarray), np.max(checkarray))
    


