from pathlib import Path
import json

import numpy as np
import pandas as pd

from helper import getUser_Interaction, getUser_pred

def load_json_array(path: Path) -> np.ndarray:
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, list):
		raise ValueError(f"Expected JSON list in {path}, got {type(data).__name__}")
	return np.asarray(data, dtype=float).reshape(-1)


def main() -> None:
	repo_root = Path(__file__).resolve().parent.parent
	conflict_dir = repo_root / "conflict"

	cu_all = np.asarray(np.load(conflict_dir / "cu.npy"), dtype=float).reshape(-1)
	list_u = np.asarray(np.load(conflict_dir / "listU.npy"), dtype=int).reshape(-1)
	scope_scores = load_json_array(conflict_dir / "SCOPE.json")
	lgcn_scores = load_json_array(conflict_dir / "LGCN.json")

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
	min_discrepancy = 0.5
	min_gap = 0.2
	mask = (discrepancy > min_discrepancy) & (score_gap > min_gap)
	candidate_idx = np.where(mask)[0]

	if candidate_idx.size == 0:
		print("No users satisfy discrepancy > 0.5 and SCOPE-LGCN gap > 0.2")
		return

	order = np.lexsort((-discrepancy[candidate_idx], -score_gap[candidate_idx]))
	ranked_idx = candidate_idx[order][:10]

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
	high_ratio = 0.01
	num_high = int(len(sorted_items) * high_ratio)

	# lấy top 10%
	popular_items = set([
		item for item, degree in sorted_items[:num_high]
	])
	print("Top users (max 10):")
	print("user_id\tcu\tdiscrepancy(1-cu)\tscope\tlgcn\tgap")
	for idx in ranked_idx:
		user_id = int(list_u[idx])
		print(
			f"{user_id}\t{cu_selected[idx]:.6f}\t{discrepancy[idx]:.6f}\t"
			f"{scope_scores[idx]:.6f}\t{lgcn_scores[idx]:.6f}\t{score_gap[idx]:.6f}"
		)
	for idx in ranked_idx:
		user_id = int(list_u[idx])
		u_is = user_interactions[user_id]
		# print(u_is) and print how many of them are popular
		popular_count = sum(1 for item in u_is if item in popular_items)
		print(f"User {user_id} interacted with {len(u_is)} items, {popular_count} of which are popular.")
		# print all interacted items
		print(f"Interacted items: {u_is}")
		# print all interacted items that are popular
		popular_interactions = [item for item in u_is if item in popular_items]
		print(f"Popular interacted items: {popular_interactions}")
		u_pred = user_test.get(user_id, [])
		print("*" * 40)
		print(f"User {user_id} has {len(u_pred)} test interactions.")
		print(f"Test interacted items: {u_pred}")
		print("#" * 60)

if __name__ == "__main__":
	main()
