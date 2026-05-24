from pathlib import Path
import json

import numpy as np


def load_npy_1d(path: Path) -> np.ndarray:
	array = np.load(path)
	return np.asarray(array).reshape(-1)


def load_score_values(path: Path) -> np.ndarray:
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	if isinstance(data, list):
		return np.asarray(data, dtype=float).reshape(-1)

	if isinstance(data, dict):
		# Preserve numeric key order if keys are user indices as strings.
		try:
			ordered_items = [value for _, value in sorted(data.items(), key=lambda x: int(x[0]))]
		except ValueError:
			ordered_items = list(data.values())
		return np.asarray(ordered_items, dtype=float).reshape(-1)

	raise ValueError(f"Unsupported score file format in {path}. Expected list or dict JSON.")


def group_users_by_ordered_values(cu_values: np.ndarray) -> dict[str, np.ndarray]:
	"""Split users by ordered cu values into requested percentile ranges.

	Groups:
	- 1-10%: lowest values
	- 11-91%: middle values
	- 91-100%: highest values
	"""
	flat_values = np.asarray(cu_values).reshape(-1)
	n_users = flat_values.shape[0]

	if n_users == 0:
		return {
			"all_users": np.array([], dtype=int),
			"lowest_1_10_percent": np.array([], dtype=int),
			"middle_11_91_percent": np.array([], dtype=int),
			"highest_91_100_percent": np.array([], dtype=int),
		}

	sorted_user_idx = np.argsort(flat_values)

	# Index cutoffs from percentages over ordered users.
	low_end = int(np.ceil(0.10 * n_users))
	mid_end = int(np.ceil(0.91 * n_users))

	low_group = sorted_user_idx[:low_end]
	mid_group = sorted_user_idx[low_end:mid_end]
	high_group = sorted_user_idx[mid_end:]

	return {
		"all_users": np.arange(n_users, dtype=int),
		"lowest_1_10_percent": low_group,
		"middle_11_91_percent": mid_group,
		"highest_91_100_percent": high_group,
	}


def main() -> None:
	repo_root = Path(__file__).resolve().parent.parent
	data_root = repo_root / "conflict"
	listu_path = data_root / "listU.npy"
	cu_path = data_root / "cu.npy"
	# score_paths = [data_root / f"{i}.json" for i in range(0, 5)]
	# score path in [LGCN, DUALCF, SCOPE]
	score_paths = [
		data_root / "LGCN.json",
		data_root / "DUALCF.json",
		data_root / "SCOPE.json",
	]
	print(score_paths)

	user_ids = load_npy_1d(listu_path).astype(int)
	cu_values = load_npy_1d(cu_path)
	cu_values = 1 - cu_values[user_ids]  # Align cu values with user IDs if needed.

	if user_ids.shape[0] != cu_values.shape[0]:
		raise ValueError(
			"Count mismatch: "
			f"users={user_ids.shape[0]}, cu={cu_values.shape[0]}"
		)

	groups = group_users_by_ordered_values(cu_values)
	group_order = ["all_users", "lowest_1_10_percent", "middle_11_91_percent", "highest_91_100_percent"]

	group_value_list_all = []
	group_value_list_1_10 = []
	group_value_list_11_91 = []
	group_value_list_91_100 = []

	print(f"Loaded users: {user_ids.shape[0]}")
	for score_path in score_paths:
		score_values = load_score_values(score_path)

		if score_values.shape[0] != user_ids.shape[0]:
			raise ValueError(
				"Count mismatch: "
				f"file={score_path.name}, users={user_ids.shape[0]}, cu={cu_values.shape[0]}, "
				f"scores={score_values.shape[0]}"
			)

		print(f"\nFile: {score_path.name}")
		group_means = {}
		for group_name in group_order:
			indices = groups[group_name]
			group_scores = score_values[indices]
			mean_score = float(np.mean(group_scores)) if indices.size > 0 else float("nan")
			group_means[group_name] = mean_score
			print(f"{group_name}: users={indices.size}, mean_score={mean_score:.6f}")

		group_value_list_all.append(group_means["all_users"])
		group_value_list_1_10.append(group_means["lowest_1_10_percent"])
		group_value_list_11_91.append(group_means["middle_11_91_percent"])
		group_value_list_91_100.append(group_means["highest_91_100_percent"])

	print("\nGroup value lists across files 0.json to 10.json:")
	print(f"all_users values: {group_value_list_all}")
	print(f"lowest_1_10_percent values: {group_value_list_1_10}")
	print(f"middle_11_91_percent values: {group_value_list_11_91}")
	print(f"highest_91_100_percent values: {group_value_list_91_100}")


if __name__ == "__main__":
	main()
