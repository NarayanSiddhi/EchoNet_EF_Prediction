import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    faiss = None


def main():
    parser = argparse.ArgumentParser(description="Build FAISS cosine index from embeddings")
    parser.add_argument("--emb_dir", type=str, required=True)
    parser.add_argument("--index_out", type=str, required=True)
    parser.add_argument("--map_out", type=str, required=True)
    args = parser.parse_args()

    emb_dir = Path(args.emb_dir)
    vecs: List[np.ndarray] = []
    names: List[str] = []

    for npy in sorted(emb_dir.glob("*.npy")):
        v = np.load(str(npy))
        if v.ndim == 1:
            v = v[None, :]
        vecs.append(v.astype(np.float32))
        names.append(npy.stem)

    if len(vecs) == 0:
        print("No embeddings found.")
        return

    X = np.concatenate(vecs, axis=0).astype(np.float32)  # [N, D]
    # ensure unit norm for cosine similarity
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / norms

    if faiss is None:
        raise RuntimeError("faiss is not installed. Please install faiss-cpu or faiss-gpu.")

    d = X.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product on unit-norm vectors = cosine similarity
    index.add(X)
    faiss.write_index(index, args.index_out)

    id_map = {i: names[i] for i in range(len(names))}
    with open(args.map_out, "w") as f:
        json.dump(id_map, f, indent=2)

    print(f"Indexed {len(names)} vectors of dim {d}.")
    print(f"Saved index to {args.index_out}")
    print(f"Saved id map to {args.map_out}")


if __name__ == "__main__":
    main()

