import json
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


OUT = Path("mnist_embeddings.json")
SAMPLES_PER_DIGIT = 35
RANDOM_STATE = 42

# Grades de parametros expostas na simulacao (cada valor vira um botao no HTML).
KPCA_GAMMAS = [0.002, 0.006, 0.018, 0.05]
TSNE_PERPLEXITIES = [5, 15, 30, 50]


def normalize(points):
    points = np.asarray(points, dtype=float)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.where(maxs - mins == 0, 1, maxs - mins)
    return ((points - mins) / span * 2 - 1).round(5)


def balanced_subset(x, y):
    rng = np.random.default_rng(RANDOM_STATE)
    selected = []
    labels = np.asarray(y).astype(int)
    for digit in range(10):
        idx = np.flatnonzero(labels == digit)
        chosen = rng.choice(idx, size=SAMPLES_PER_DIGIT, replace=False)
        selected.extend(chosen.tolist())
    rng.shuffle(selected)
    return x[selected], labels[selected]


def main():
    print("Baixando MNIST do OpenML (so na primeira vez)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    x, y = balanced_subset(mnist.data.astype(np.float32), mnist.target)
    x01 = x / 255.0

    scaler = StandardScaler(with_mean=True, with_std=True)
    x_scaled = scaler.fit_transform(x01)

    pca_model = PCA(n_components=50, random_state=RANDOM_STATE)
    x_pca50 = pca_model.fit_transform(x_scaled)
    pca2 = normalize(x_pca50[:, :2])

    embeddings = {
        "pca": pca2.tolist(),
        "kpca": {},
        "tsne": {},
    }

    for gamma in KPCA_GAMMAS:
        print(f"KernelPCA gamma={gamma}")
        model = KernelPCA(
            n_components=2,
            kernel="rbf",
            gamma=gamma,
            eigen_solver="arpack",
            random_state=RANDOM_STATE,
        )
        embeddings["kpca"][str(gamma)] = normalize(model.fit_transform(x_pca50)).tolist()

    for perplexity in TSNE_PERPLEXITIES:
        print(f"t-SNE perplexity={perplexity}")
        model = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=900,
            random_state=RANDOM_STATE,
        )
        embeddings["tsne"][str(perplexity)] = normalize(model.fit_transform(x_pca50)).tolist()

    payload = {
        "meta": {
            "source": "MNIST from OpenML mnist_784",
            "original_dimensions": 784,
            "image_shape": [28, 28],
            "samples": int(len(y)),
            "samples_per_digit": SAMPLES_PER_DIGIT,
            "random_state": RANDOM_STATE,
            "explained_variance_first_2": float(pca_model.explained_variance_ratio_[:2].sum()),
            "explained_variance_first_50": float(pca_model.explained_variance_ratio_[:50].sum()),
        },
        "labels": y.astype(int).tolist(),
        "pixels": np.rint(x01 * 15).astype(np.uint8).tolist(),
        "embeddings": embeddings,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Pronto: {OUT} com {len(y)} digitos do MNIST")
    print("Agora rode:  python embed_data.py")


if __name__ == "__main__":
    main()
