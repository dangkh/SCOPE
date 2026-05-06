import argparse
import os

import numpy as np
from sklearn.neighbors import NearestNeighbors


def resolve_input_path(dataset, input_path):
    if input_path is not None:
        return input_path

    if dataset is not None:
        dataset_dir = os.path.join("data", dataset)
        candidates = [
            os.path.join(dataset_dir, "user_rep.npy"),
            os.path.join(dataset_dir, "_user_rep.npy"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    return "user_rep.npy"


def resolve_output_path(dataset, output_path, input_path, top_k):
    if output_path is not None:
        return output_path

    if dataset is not None:
        return os.path.join("data", dataset, f"user_top{top_k}user.npy")

    input_dir = os.path.dirname(input_path)
    return os.path.join(input_dir, f"user_top{top_k}user.npy")


def build_user_user_knn(user_rep, top_k=10, metric="cosine", return_scores=False):
    user_rep = np.asarray(user_rep, dtype=np.float32)
    if user_rep.ndim != 2:
        raise ValueError(f"user_rep must be a 2D array, got shape {user_rep.shape}")

    num_users = user_rep.shape[0]
    if top_k >= num_users:
        raise ValueError(f"top_k must be < num_users (top_k={top_k}, num_users={num_users})")

    nn = NearestNeighbors(n_neighbors=top_k + 1, metric=metric, algorithm="auto")
    nn.fit(user_rep)
    distances, indices = nn.kneighbors(user_rep, return_distance=True)

    topk_users = np.empty((num_users, top_k), dtype=np.int32)
    topk_distances = np.empty((num_users, top_k), dtype=np.float32)
    for user_id in range(num_users):
        keep = indices[user_id] != user_id
        topk_users[user_id] = indices[user_id][keep][:top_k]
        topk_distances[user_id] = distances[user_id][keep][:top_k]

    if not return_scores:
        return topk_users

    if metric == "cosine":
        topk_scores = 1.0 - topk_distances
    else:
        topk_scores = topk_distances
    return topk_users, topk_scores.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Build top-k similar users from user_rep.npy.")
    parser.add_argument("--dataset", "-d", type=str, default="book", help="dataset name under ./data/")
    parser.add_argument("--input", "-i", type=str, default=None, help="path to user_rep.npy")
    parser.add_argument("--output", "-o", type=str, default=None, help="path to save top-k user indices")
    parser.add_argument("--k", type=int, default=10, help="number of similar users for each user")
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean"],
        help="similarity/distance metric",
    )
    parser.add_argument(
        "--save_scores",
        action="store_true",
        help="also save scores to <output basename>_scores.npy",
    )
    args = parser.parse_args()

    input_path = resolve_input_path(args.dataset, args.input)
    output_path = resolve_output_path(args.dataset, args.output, input_path, args.k)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"user representation file not found: {input_path}")

    user_rep = np.load(input_path, allow_pickle=True)
    result = build_user_user_knn(
        user_rep,
        top_k=args.k,
        metric=args.metric,
        return_scores=args.save_scores,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if args.save_scores:
        topk_users, topk_scores = result
        np.save(output_path, topk_users)
        score_path = os.path.splitext(output_path)[0] + "_scores.npy"
        np.save(score_path, topk_scores)
        print(f"Saved top-{args.k} user indices to: {output_path}")
        print(f"Saved top-{args.k} user scores to: {score_path}")
    else:
        np.save(output_path, result)
        print(f"Saved top-{args.k} user indices to: {output_path}")

    print(f"Input shape: {user_rep.shape}")
    print(f"Output shape: {result[0].shape if args.save_scores else result.shape}")


if __name__ == "__main__":
    main()
