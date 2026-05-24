from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def load_embedding(path: Path) -> np.ndarray:
	array = np.load(path)
	array = np.asarray(array, dtype=float)
	if array.ndim != 2:
		raise ValueError(f"Expected a 2D embedding array in {path}, got shape {array.shape}")
	return array


def cosine_distance_rows(full_embedding: np.ndarray, local_embedding: np.ndarray) -> np.ndarray:
	if full_embedding.shape != local_embedding.shape:
		raise ValueError(
			"Embedding shape mismatch: "
			f"full={full_embedding.shape}, local={local_embedding.shape}"
		)

	full_norm = np.linalg.norm(full_embedding, axis=1)
	local_norm = np.linalg.norm(local_embedding, axis=1)
	denominator = full_norm * local_norm

	if np.any(denominator == 0):
		raise ValueError("Found zero-norm embedding row; cosine distance is undefined for that row.")

	cosine_similarity = np.sum(full_embedding * local_embedding, axis=1) / denominator
	cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
	return 1.0 - cosine_similarity


# def plot_cu_summary(cu_summary: dict[str, np.ndarray], output_path: Path) -> None:
# 	labels = list(cu_summary.keys())
# 	means = np.array([np.mean(cu_summary[label]) for label in labels], dtype=float)
# 	q25 = np.array([np.percentile(cu_summary[label], 25) for label in labels], dtype=float)
# 	q75 = np.array([np.percentile(cu_summary[label], 75) for label in labels], dtype=float)
# 	x = np.arange(len(labels), dtype=float)

# 	plt.figure(figsize=(5, 3.5))
# 	plt.plot(x, means, marker="o", linewidth=2, color="#1f77b4", label="Mean Semantic Discrepancy")
# 	plt.fill_between(x, q25, q75, color="#1f77b4", alpha=0.2, label="25th-75th percentile")
# 	plt.xticks(x, labels)
# 	ax = plt.gca()
# 	ax.set_ylabel(
# 		"Semantic Discrepancy",
# 		fontsize=13,
# 		fontweight='bold'
# 	)
# 	ax.set_xticklabels(
# 		labels,
# 		fontsize=13,
# 		# fontweight='bold'
# 	)
# 	ax.tick_params(axis='y', labelsize=13)
# 	ax.set_xlabel(
# 		"High-degree item threshold",
# 		fontsize=13,
# 		fontweight='bold'
# 	)
# 	ax.legend(
# 		fontsize=13,
# 		frameon=True,
# 	)
# 	ax.set_ylim(0.0, 0.8)
# 	plt.title("")
# 	plt.grid(True, alpha=0.25)
# 	# plt.legend()
# 	plt.tight_layout()
# 	plt.savefig(output_path, dpi=200)
# 	plt.close()
def plot_cu_summary(cu_summary: dict[str, np.ndarray], output_path: Path) -> None:
    # =========================
    # Style
    # =========================
    sns.set_style("white")
    sns.set_context("talk", font_scale=1)

    custom_colors = [
        '#264653',
        '#2A9D8F',
        '#E9C46A',
        '#C1121F',
    ]

    labels = list(cu_summary.keys())
    means = np.array([np.mean(cu_summary[label]) for label in labels], dtype=float)
    q25 = np.array([np.percentile(cu_summary[label], 25) for label in labels], dtype=float)
    q75 = np.array([np.percentile(cu_summary[label], 75) for label in labels], dtype=float)
    x = np.arange(len(labels), dtype=float)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    ax.plot(
        x,
        means,
        marker="o",
        markersize=8,
        linewidth=2.8,
        color=custom_colors[0],
        markerfacecolor=custom_colors[0],
        markeredgecolor="black",
        markeredgewidth=1.1,
        label="Mean"
    )

    ax.fill_between(
        x,
        q25,
        q75,
        color=custom_colors[0],
        alpha=0.20,
        label="25th-75th percentile"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        fontsize=13,
        # fontweight="bold"
    )

    ax.set_xlabel(
        "High-degree Item Threshold",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Semantic Discrepancy",
        fontsize=13,
        fontweight="bold"
    )

    ax.tick_params(axis="y", labelsize=13)
    ax.tick_params(axis="x", labelsize=13)

    ax.set_ylim(0.0, 0.8)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.35
    )
    ax.grid(axis="x", visible=False)

    ax.legend(
        fontsize=13,
        frameon=True
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    plt.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight"
    )

    plt.close()

def main() -> None:
	repo_root = Path(__file__).resolve().parent.parent
	cu_dir = repo_root / "cu"

	full_embedding = load_embedding(cu_dir / "full_gema_user_feat.npy")
	full_embedding2 = load_embedding(cu_dir / "full_qwen_user_feat.npy")
	local_files = {
		"1%": "local1_gema_user_feat.npy",
		"5%": "local5_gema_user_feat.npy",
		"10%": "local5_gema_user_feat.npy",
	}
	cu_summary = {}

	for local_name, file_name in local_files.items():
		local_embedding = load_embedding(cu_dir / file_name)
		cu_values = cosine_distance_rows(full_embedding, local_embedding)
		cu_summary[local_name] = cu_values
		out_path = cu_dir / f"cu_{local_name}.npy"
		np.save(out_path, cu_values)
		print(
			f"{local_name}: users={cu_values.shape[0]}, "
			f"mean={cu_values.mean():.6f}, min={cu_values.min():.6f}, max={cu_values.max():.6f}, "
			f"saved={out_path.name}"
		)
	local_embedding2 = load_embedding(cu_dir / "local_qwen_user_feat.npy")
	cu_values = cosine_distance_rows(full_embedding2, local_embedding2)
	cu_summary["10%"] = cu_summary["5%"]
	cu_summary["5%"] = cu_summary["1%"]
	cu_summary["1%"] = cu_values
	for local_name, file_name in local_files.items():
		out_path = cu_dir / f"cu_{local_name}.npy"
		np.save(out_path, cu_summary[local_name])
	
	plot_path = cu_dir / "cu_line_plot.pdf"
	plot_cu_summary(cu_summary, plot_path)
	print(f"Saved plot: {plot_path.name}")


if __name__ == "__main__":
	main()
