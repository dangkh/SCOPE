import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_array(path: Path) -> np.ndarray:
	array = np.load(path)
	return np.asarray(array).reshape(-1)


def main() -> None:
	parser = argparse.ArgumentParser(description="Scatter plot of cu vs alpha and Pearson correlation.")
	repo_root = Path(__file__).resolve().parent.parent
	parser.add_argument("--cu", type=Path, default=repo_root / "cu.npy", help="Path to cu.npy")
	parser.add_argument("--alpha", type=Path, default=repo_root / "alpha.npy", help="Path to alpha.npy")
	parser.add_argument("--out", type=Path, default=repo_root / "scatter_cu_alpha.png", help="Output image path")
	args = parser.parse_args()

	cu = load_array(args.cu)
	alpha = load_array(args.alpha)

	if cu.shape[0] != alpha.shape[0]:
		raise ValueError(f"Shape mismatch: cu {cu.shape} vs alpha {alpha.shape}")

	corr = np.corrcoef(cu, alpha)[0, 1]
	print(f"Pearson correlation (cu, alpha): {corr:.6f}")

	plt.figure(figsize=(7, 6))
	plt.scatter(cu, alpha, s=18, alpha=0.6, edgecolors="none")
	plt.title(f"cu vs alpha  (r = {corr:.4f})")
	plt.xlabel("cu")
	plt.ylabel("alpha")
	plt.grid(True, alpha=0.25)
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
