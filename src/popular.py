import argparse
import numpy as np
import pandas as pd
from helper import build_item_item_knn, get_itemDesc, getUser_Interaction
import json



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='book', help='name of datasets')
    args, _ = parser.parse_known_args()
    print(args)
    file_path = f'./data/{args.dataset}/{args.dataset}.inter'
    interDF = pd.read_csv(file_path, sep="\t", usecols=['userID', 'itemID', 'x_label'])
    interDF['userID'] = interDF['userID'].astype(int)
    interDF['itemID'] = interDF['itemID'].astype(int)
    user_interactions = getUser_Interaction(interDF)
    # from user_interaction, get item degree
    item_degree = {}
    for u, items in user_interactions.items():
        for item in items:
            if item not in item_degree:
                item_degree[item] = 0
            item_degree[item] += 1
    
    sorted_items = sorted(item_degree.items(), key=lambda x: x[1], reverse=False)
    total_items = len(sorted_items)
    
    # Split all items into 4 percentile ranges
    # 0-30%: most popular items
    # 31-60%: moderately popular items
    # 61-90%: less popular items
    # 91-100%: least popular items
    idx_30 = int(total_items * 0.30)
    idx_60 = int(total_items * 0.60)
    idx_90 = int(total_items * 0.90)
    
    groups = {
        "0-30%": sorted_items[0:idx_30],
        "31-60%": sorted_items[idx_30:idx_60],
        "61-90%": sorted_items[idx_60:idx_90],
        "91-100%": sorted_items[idx_90:],
    }
    
    print("\n" + "="*70)
    print("Item counts by degree range (percentile split)")
    print("="*70)
    for range_name, items in groups.items():
        count = len(items)
        if items:
            min_degree = items[-1][1]  # Last item in group has lowest degree
            max_degree = items[0][1]   # First item in group has highest degree
            print(f"Range {range_name}: {count:5d} items | Degree range: [{min_degree:4d}, {max_degree:4d}]")
        else:
            print(f"Range {range_name}: {count:5d} items")
    print("="*70)
    print(f"Total items: {total_items}")
    print("="*70)
    # load file GLORIA-book-idx21-top20.json
    with open('./recommend_topk/hitGLORIA-book-idx192-top20.json', 'r') as f:
        itemPredDict = json.load(f)
    with open('./recommend_topk/recallGLORIA-book-idx192-top20.json', 'r') as f:
        itemRecallDict = json.load(f)
        
    # get the hit of items in itemPredDict
    print("\n" + "="*70)
    print("Degree distribution of predicted items by percentile group")
    print("="*70)
    for range_name, items in groups.items():    
        group_item_ids = set(int(item[0]) for item in items)
        group_pred_hit = [itemPredDict[item] for item in itemPredDict.keys() if int(item) in group_item_ids]
        group_pred_recall = [itemRecallDict[item] for item in itemRecallDict.keys() if int(item) in group_item_ids]
        
        if group_pred_hit:
            # hit = sum group predicted items in this group
            hit = sum(group_pred_hit)
            rec = sum(group_pred_recall)
            print(f"Range {range_name}: {hit} predicted items , recall sum: {rec:.4f}")
        else:
            print(f"Range {range_name}: 0 predicted items")
    print("="*70)