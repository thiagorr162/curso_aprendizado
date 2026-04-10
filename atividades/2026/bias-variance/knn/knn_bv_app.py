from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle

X0 = np.array([0.5, 0.5])
GRID_SIZE = 40


# --- funcoes verdadeiras -----------------------------------------------

def f_true(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return (
        np.sin(np.pi * np.asarray(x1, dtype=float))
        * np.sin(np.pi * np.asarray(x2, dtype=float))
    )


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def boundary(x1: np.ndarray) -> np.ndarray:
    """Fronteira polinomial verdadeira: y=1 acima, y=0 abaixo."""
    x1 = np.asarray(x1, dtype=float)
    return 0.15 + 0.7 * x1 - 0.65 * x1 ** 2


def f_prob(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """P(y=1|x): quasi-deterministica acima/abaixo da fronteira + ruido de label."""
    return sigmoid((np.asarray(x2, dtype=float) - boundary(x1)) / 0.06)


# --- KNN helpers -------------------------------------------------------

def precompute_sorted_neighbors(
    x_train: np.ndarray, y_train: np.ndarray, grid_pts: np.ndarray
) -> np.ndarray:
    """(m, n) — y dos vizinhos ordenados por distancia para cada ponto do grid."""
    dists = np.linalg.norm(grid_pts[:, None, :] - x_train[None, :, :], axis=2)
    return y_train[np.argsort(dists, axis=1)]


# --- decomposicoes -----------------------------------------------------

def compute_decomposition_regression(
    n: int, sigma: float, n_sim: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f0 = float(f_true(np.array([X0[0]]), np.array([X0[1]]))[0])
    preds = np.empty((n_sim, n))
    for i in range(n_sim):
        rng = np.random.default_rng(seed + i * 7919)
        x_tr = rng.uniform(0.0, 1.0, (n, 2))
        y_tr = f_true(x_tr[:, 0], x_tr[:, 1]) + rng.normal(0.0, sigma, n)
        sorted_y = y_tr[np.argsort(np.linalg.norm(x_tr - X0, axis=1))]
        preds[i] = np.cumsum(sorted_y) / np.arange(1, n + 1)
    ks = np.arange(1, n + 1)
    bias2 = (preds.mean(axis=0) - f0) ** 2
    variance = preds.var(axis=0)
    irred = np.full(n, sigma ** 2)
    return ks, bias2, variance, irred, preds


def compute_decomposition_classification(
    n: int, n_sim: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p0 = float(f_prob(np.array([X0[0]]), np.array([X0[1]]))[0])
    preds = np.empty((n_sim, n))
    for i in range(n_sim):
        rng = np.random.default_rng(seed + i * 7919)
        x_tr = rng.uniform(0.0, 1.0, (n, 2))
        p_tr = f_prob(x_tr[:, 0], x_tr[:, 1])
        y_tr = (rng.uniform(size=n) < p_tr).astype(float)
        sorted_y = y_tr[np.argsort(np.linalg.norm(x_tr - X0, axis=1))]
        preds[i] = np.cumsum(sorted_y) / np.arange(1, n + 1)
    ks = np.arange(1, n + 1)
    bias2 = (preds.mean(axis=0) - p0) ** 2
    variance = preds.var(axis=0)
    irred = np.full(n, p0 * (1.0 - p0))  # erro de Bayes
    return ks, bias2, variance, irred, preds


# --- app ---------------------------------------------------------------

class KNNExplorer:
    N_SIM = 300
    SEED = 42

    def __init__(self) -> None:
        self.n = 100
        self.sigma = 0.3
        self.dataset_seed = 0
        self.mode = "regression"  # ou "classification"

        g = np.linspace(0.0, 1.0, GRID_SIZE)
        G1, G2 = np.meshgrid(g, g)
        self.true_heatmap_reg = f_true(G1, G2)
        self.true_heatmap_cls = f_prob(G1, G2)
        self.grid_pts = np.column_stack([G1.ravel(), G2.ravel()])

        self.fig = plt.figure(figsize=(15, 7), constrained_layout=False)
        self.fig.canvas.manager.set_window_title("KNN Bias-Variance 2D")

        gs = self.fig.add_gridspec(
            5, 13, left=0.05, right=0.97, top=0.91, bottom=0.06, hspace=1.0, wspace=1.0
        )
        self.ax_true = self.fig.add_subplot(gs[:4, 0:4])
        self.ax_knn  = self.fig.add_subplot(gs[:4, 4:8])
        self.ax_bv   = self.fig.add_subplot(gs[:4, 9:])

        self.sl_k = Slider(
            self.fig.add_subplot(gs[4, 2:10]), "k (vizinhos)", 1, self.n,
            valinit=5, valstep=1,
        )
        self.btn_mode = Button(
            self.fig.add_subplot(gs[4, 11:13]), "Classificacao",
            color="#e8f4e8", hovercolor="#c8e8c8",
        )

        self.title_artist = self.fig.suptitle("", fontsize=15, fontweight="bold")

        self._recompute()
        self._new_display_dataset()
        self.sl_k.on_changed(self._on_k_change)
        self.btn_mode.on_clicked(self._on_toggle_mode)
        self._update_title()
        self.redraw()

    # ------------------------------------------------------------------
    def _update_title(self) -> None:
        mode_str = "Regressao" if self.mode == "regression" else "Classificacao"
        self.title_artist.set_text(
            f"Trade-off vies-variancia com KNN 2D — {mode_str}  (x₀ = (0.5, 0.5))"
        )
        self.btn_mode.label.set_text(
            "→ Classificacao" if self.mode == "regression" else "→ Regressao"
        )

    def _recompute(self) -> None:
        if self.mode == "regression":
            self.ks, self.bias2, self.variance, self.irred, self.preds = (
                compute_decomposition_regression(self.n, self.sigma, self.N_SIM, self.SEED)
            )
        else:
            self.ks, self.bias2, self.variance, self.irred, self.preds = (
                compute_decomposition_classification(self.n, self.N_SIM, self.SEED)
            )
        self.total = self.bias2 + self.variance + self.irred

    def _new_display_dataset(self) -> None:
        rng = np.random.default_rng(self.dataset_seed)
        self.x_train = rng.uniform(0.0, 1.0, (self.n, 2))
        if self.mode == "regression":
            self.y_train = (
                f_true(self.x_train[:, 0], self.x_train[:, 1])
                + rng.normal(0.0, self.sigma, self.n)
            )
        else:
            p_tr = f_prob(self.x_train[:, 0], self.x_train[:, 1])
            self.y_train = (rng.uniform(size=self.n) < p_tr).astype(float)

        self.sorted_y_grid = precompute_sorted_neighbors(
            self.x_train, self.y_train, self.grid_pts
        )
        dists_x0 = np.linalg.norm(self.x_train - X0, axis=1)
        order_x0 = np.argsort(dists_x0)
        self.sorted_dists_x0 = dists_x0[order_x0]
        self.sorted_x0 = self.x_train[order_x0]

    def _on_k_change(self, _: float) -> None:
        self._draw_knn()
        self._draw_bv()
        self.fig.canvas.draw_idle()

    def _on_toggle_mode(self, _: object) -> None:
        self.mode = "classification" if self.mode == "regression" else "regression"
        self._recompute()
        self._new_display_dataset()
        self._update_title()
        self._draw_true()
        self._draw_knn()
        self._draw_bv()
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    def redraw(self) -> None:
        self._draw_true()
        self._draw_knn()
        self._draw_bv()
        self.fig.canvas.draw_idle()

    def _heatmap_cfg(self) -> tuple[np.ndarray, float, float, str, str]:
        if self.mode == "regression":
            hmap = self.true_heatmap_reg
            vmin, vmax = float(hmap.min()), float(hmap.max())
            cmap = "RdYlBu_r"
            clabel = "f(x)"
        else:
            hmap = self.true_heatmap_cls
            vmin, vmax = 0.0, 1.0
            cmap = "RdBu_r"
            clabel = "P(y=1|x)"
        return hmap, vmin, vmax, cmap, clabel

    def _draw_true(self) -> None:
        ax = self.ax_true
        ax.clear()
        hmap, vmin, vmax, cmap, _ = self._heatmap_cfg()

        if self.mode == "regression":
            ax.imshow(hmap, origin="lower", extent=[0, 1, 0, 1],
                      cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.scatter(self.x_train[:, 0], self.x_train[:, 1],
                       c=self.y_train, cmap=cmap, vmin=vmin, vmax=vmax,
                       edgecolors="k", linewidths=0.4, s=40, zorder=3)
            title = "f(x) verdadeira + dados"
        else:
            # regioes binarias com alpha leve + fronteira polinomial explicita
            ax.imshow((hmap > 0.5).astype(float), origin="lower", extent=[0, 1, 0, 1],
                      cmap="RdBu_r", vmin=0, vmax=1, aspect="auto", alpha=0.3)
            x1_line = np.linspace(0, 1, 400)
            ax.plot(x1_line, boundary(x1_line), color="k", linewidth=2.5,
                    label="fronteira verdadeira", zorder=4)
            colors = np.where(self.y_train == 1, "#c0392b", "#2980b9")
            ax.scatter(self.x_train[:, 0], self.x_train[:, 1],
                       color=colors, edgecolors="k", linewidths=0.4, s=40, zorder=3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(loc="upper right", fontsize=8)
            title = "Fronteira verdadeira + dados (com ruido)"

        ax.scatter(*X0, marker="*", s=250, color="white", edgecolors="k", linewidths=1.2, zorder=5)
        ax.set_title(title)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")

    def _draw_knn(self) -> None:
        k = int(self.sl_k.val)
        ax = self.ax_knn
        ax.clear()
        _, vmin, vmax, cmap, _ = self._heatmap_cfg()

        pred_flat = self.sorted_y_grid[:, :k].mean(axis=1)
        pred_map = pred_flat.reshape(GRID_SIZE, GRID_SIZE)
        ax.imshow(pred_map, origin="lower", extent=[0, 1, 0, 1],
                  cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        if self.mode == "classification":
            # regioes de decisao binarias (mais rapido que contour)
            ax.imshow((pred_map > 0.5).astype(float), origin="lower", extent=[0, 1, 0, 1],
                      cmap="RdBu_r", vmin=0, vmax=1, aspect="auto", alpha=0.5)

        if self.mode == "regression":
            ax.scatter(self.x_train[:, 0], self.x_train[:, 1],
                       c=self.y_train, cmap=cmap, vmin=vmin, vmax=vmax,
                       edgecolors="k", linewidths=0.4, s=25, zorder=3, alpha=0.6)
        else:
            colors = np.where(self.y_train == 1, "#c0392b", "#2980b9")
            ax.scatter(self.x_train[:, 0], self.x_train[:, 1],
                       color=colors, edgecolors="k", linewidths=0.4, s=25, zorder=3, alpha=0.6)

        ax.scatter(self.sorted_x0[:k, 0], self.sorted_x0[:k, 1],
                   s=80, edgecolors="white", linewidths=1.5, facecolors="none", zorder=4)
        radius = float(self.sorted_dists_x0[k - 1])
        ax.add_patch(Circle(X0, radius, fill=False, edgecolor="white",
                            linewidth=1.5, linestyle="--", zorder=4))
        ax.scatter(*X0, marker="*", s=250, color="white", edgecolors="k", linewidths=1.2, zorder=5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Predicao KNN (k={k})")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")

    def _draw_bv(self) -> None:
        k = int(self.sl_k.val)
        ax = self.ax_bv
        ax.clear()

        irred_label = "irredutivel (σ²)" if self.mode == "regression" else "irredutivel (p·(1−p))"
        ax.plot(self.ks, self.bias2,    color="#2a7f9e", linewidth=2.5, label="vies²")
        ax.plot(self.ks, self.variance, color="#2b8b67", linewidth=2.5, label="variancia")
        ax.plot(self.ks, self.irred,    color="#d69a1e", linewidth=2.5, label=irred_label)
        ax.plot(self.ks, self.total,    color="#7f4acb", linewidth=3.2, label="total")

        ax.axvline(k, color="#c55f3b", linestyle="--", linewidth=2, alpha=0.8)
        ax.scatter([k], [self.total[k - 1]], color="#c55f3b", s=70, zorder=5)
        best_k = int(self.ks[np.argmin(self.total)])
        ax.axvline(best_k, color="#aaa", linestyle=":", linewidth=1.5, alpha=0.7)

        ax.set_xlabel("k")
        ax.set_ylabel("erro esperado em x₀")
        ax.set_title("Decomposicao do erro vs k")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.18)
        ax.set_xlim(1, len(self.ks))
        ax.text(
            0.02, 0.97,
            f"k={k}:  vies²={self.bias2[k-1]:.4f}  var={self.variance[k-1]:.4f}  "
            f"total={self.total[k-1]:.4f}\nmelhor k={best_k}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ede5fa", edgecolor="none"),
        )


def main() -> None:
    KNNExplorer()
    plt.show()


if __name__ == "__main__":
    main()
