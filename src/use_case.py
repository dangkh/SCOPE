from pathlib import Path
import json

import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from helper import getUser_Interaction, getUser_pred

def load_json_array(path: Path) -> np.ndarray:
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, list):
		raise ValueError(f"Expected JSON list in {path}, got {type(data).__name__}")
	return np.asarray(data, dtype=float).reshape(-1)
def loadJson_prf(path):
	with open(path, 'r', encoding='utf-8') as jsonfile:
		jsonfile = json.load(jsonfile)

	profile = []
	for ii in tqdm(range(len(jsonfile))):
		try:
			raw = jsonfile[str(ii)]['summary']
			clean = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.DOTALL)
			level = json.loads(clean)
			summarization = level["summarization"]
			reasoning = level["reasoning"]
		except Exception:
			summarization = jsonfile[str(ii)]['summary']
		profile.append(summarization)
	return profile

def main() -> None:
	repo_root = Path(__file__).resolve().parent.parent
	conflict_dir = repo_root / "conflict"

	cu_all = np.asarray(np.load(conflict_dir / "cu.npy"), dtype=float).reshape(-1)
	list_u = np.asarray(np.load(conflict_dir / "listU.npy"), dtype=int).reshape(-1)
	scope_scores = load_json_array(conflict_dir / "SCOPE.json")
	lgcn_scores = load_json_array(conflict_dir / "LGCN.json")
	# predScope via predict_SCOPE.npy and predLGCN via predict_LGCN.npy
	pred_scope = np.asarray(np.load(conflict_dir / "predict_SCOPE.npy"), dtype=int)
	pred_lgcn = np.asarray(np.load(conflict_dir / "predict_LGCN.npy"), dtype=int)
	# userFull_prf via full_batch_gema_urf_prf.json
	user_full_prf = loadJson_prf(conflict_dir / "full_batch_gema_usr_prf.json")
	user_local_prf = loadJson_prf(conflict_dir / "local_batch_gema_usr_prf.json")

	# assert user_full_prf and local_prf have same number of users as cu_all
	if len(user_full_prf) != cu_all.shape[0]:
		raise ValueError(
			f"User profile count mismatch: full_prf={len(user_full_prf)} vs cu_all={cu_all.shape[0]}"
		)

	if scope_scores.shape[0] != lgcn_scores.shape[0]:
		raise ValueError(
			f"SCORE length mismatch: SCOPE={scope_scores.shape[0]} vs LGCN={lgcn_scores.shape[0]}"
		)

	if list_u.shape[0] != scope_scores.shape[0]:
		raise ValueError(
			f"User/score length mismatch: listU={list_u.shape[0]} vs scores={scope_scores.shape[0]}"
		)

	if np.any(list_u < 0) or np.any(list_u >= cu_all.shape[0]):
		raise ValueError(
			"listU contains out-of-range user ids for cu.npy: "
			f"cu_size={cu_all.shape[0]}, min_user={int(list_u.min())}, max_user={int(list_u.max())}"
		)

	# Update/select cu only for users present in listU.
	cu_selected = cu_all[list_u]
	discrepancy = 1.0 - cu_selected
	score_gap = scope_scores - lgcn_scores

	# Keep users with high discrepancy and clearly better SCOPE recall.
	min_discrepancy = 0.1
	min_gap = 0
	mask = (discrepancy > min_discrepancy) & (score_gap >= min_gap)
	candidate_idx = np.where(mask)[0]

	if candidate_idx.size == 0:
		print("No users satisfy discrepancy > 0.5 and SCOPE-LGCN gap > 0.2")
		return

	order = np.lexsort((-discrepancy[candidate_idx], -score_gap[candidate_idx]))
	ranked_idx = candidate_idx[order]  # Top 100 candidates

	print(f"Candidates found: {candidate_idx.size}")
	

	dataset = 'movie'
	file_path = f'./data/{dataset}/fullMeta_{dataset}.csv'
	metaDF = pd.read_csv(file_path)
	metaDF = pd.DataFrame(metaDF)
	unique_meta_asin = set(metaDF['asin'])
	print(f"[Meta] Unique ASINs: {len(unique_meta_asin)}")

	file_path = f'./data/{dataset}/{dataset}.inter'
	interDF = pd.read_csv(file_path, sep="\t", usecols=['userID', 'itemID', 'x_label'])
	interDF['userID'] = interDF['userID'].astype(int)
	interDF['itemID'] = interDF['itemID'].astype(int)

	user_interactions = getUser_Interaction(interDF)
	user_test = getUser_pred(interDF)
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
	high_ratio = 0.1
	num_high = int(len(sorted_items) * high_ratio)

	# lấy top 10%
	popular_items = set([
		item for item, degree in sorted_items[:num_high]
	])
	# print("Top users (max 10):")
	# print("user_id\tcu\tdiscrepancy(1-cu)\tscope\tlgcn\tgap")
	for idx in ranked_idx:
		user_id = int(list_u[idx])
		print(
			f"{user_id}\t{cu_selected[idx]:.6f}\t{discrepancy[idx]:.6f}\t"
			f"{scope_scores[idx]:.6f}\t{lgcn_scores[idx]:.6f}\t{score_gap[idx]:.6f}"
		)
	allPossibleUser = []
	for idx in ranked_idx:
		user_id = int(list_u[idx])
		u_pred = user_test.get(user_id, [])
		if len(u_pred) <=1:
			# print(f"User {user_id} has no test interactions, skipping detailed analysis.")
			# print("#" * 60)
			continue
		u_is = user_interactions[user_id]
		# print(u_is) and print how many of them are popular
		popular_count = sum(1 for item in u_is if item in popular_items)
		# print(f"User {user_id} interacted with {len(u_is)} items, {popular_count} of which are popular.")
		# print all interacted items
		# print(f"Interacted items: {u_is}")
		# print all interacted items that are popular
		popular_interactions = [item for item in u_is if item in popular_items]
		# print(f"Popular interacted items: {popular_interactions}")
		# print("*" * 40)
		# print(f"User {user_id} has {len(u_pred)} test interactions.")
		# print(f"Test interacted items: {u_pred}")
		user_scope_pred = pred_scope[idx]
		user_lgcn_pred = pred_lgcn[idx]

		correct_scope = sum(1 for item in u_pred if item in user_scope_pred)
		correct_lgcn = sum(1 for item in u_pred if item in user_lgcn_pred)

		# print(f"Predicted SCOPE items: {correct_scope}")
		# print(f"Predicted LGCN items: {correct_lgcn}")
		if correct_scope > 1 and correct_lgcn == 0:
			allPossibleUser.append(user_id)
		print("#" * 60)
	print(f"Total users with >1 test interactions and predicted SCOPE items: {len(allPossibleUser)}")
	print("user: ", allPossibleUser)
	selected_users = allPossibleUser

	for idx in ranked_idx:
		user_id = int(list_u[idx])
		if user_id not in selected_users:
			continue
		u_pred = user_test.get(user_id, [])
		u_is = user_interactions[user_id]
		print(f"Detailed analysis for User {user_id}:")
		print(f"CU: {cu_selected[list_u == user_id][0]:.6f}")
		print(f"Discrepancy (1-CU): {discrepancy[list_u == user_id][0]:.6f}")
		print(f"SCOPE score: {scope_scores[list_u == user_id][0]:.6f}")
		print(f"LGCN score: {lgcn_scores[list_u == user_id][0]:.6f}")
		print(f"Score gap (SCOPE - LGCN): {score_gap[list_u == user_id][0]:.6f}")
		print(f"Interacted items: {u_is}")
		popular_interactions = [item for item in u_is if item in popular_items]
		# print title of popular items:
		for item in popular_interactions:
			title = metaDF[metaDF['asin'] == item]['title'].values
			if len(title) > 0:
				print(f"Popular item {item} title: {title[0]}")
		print(f"Popular interacted items: {popular_interactions}")
		print(f"Ground truth items: {u_pred}")
		user_scope_pred = pred_scope[idx]
		user_lgcn_pred = pred_lgcn[idx]

		correct_scope = sum(1 for item in u_pred if item in user_scope_pred)
		correct_lgcn = sum(1 for item in u_pred if item in user_lgcn_pred)

		print(f"Predicted SCOPE items: {correct_scope}")
		# print(title of predicted SCOPE items)
		for item in user_scope_pred:
			title = metaDF[metaDF['asin'] == item]['title'].values
			if len(title) > 0:
				print(f"Predicted SCOPE item {item} title: {title[0]}")
		print(f"Predicted LGCN items: {correct_lgcn}")
		# print(title of predicted LGCN items)
		for item in user_lgcn_pred:
			title = metaDF[metaDF['asin'] == item]['title'].values
			if len(title) > 0:
				print(f"Predicted LGCN item {item} title: {title[0]}")
		print(f"User full profile: {user_full_prf[user_id]}")
		print(f"User local profile: {user_local_prf[user_id]}")
		print("#" * 60)
	


if __name__ == "__main__":
	main()
