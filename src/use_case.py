from pathlib import Path
import json

import numpy as np


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
	print("Top users (max 10):")
	print("user_id\tcu\tdiscrepancy(1-cu)\tscope\tlgcn\tgap")
	for idx in ranked_idx:
		user_id = int(list_u[idx])
		print(
			f"{user_id}\t{cu_selected[idx]:.6f}\t{discrepancy[idx]:.6f}\t"
			f"{scope_scores[idx]:.6f}\t{lgcn_scores[idx]:.6f}\t{score_gap[idx]:.6f}"
		)


if __name__ == "__main__":
	main()
