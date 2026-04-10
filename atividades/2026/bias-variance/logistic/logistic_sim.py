from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ── constantes ───────────────────────────────────────────────────────────────
SEED        = 42
INIT_THRESH = 0.5
GRID_RES    = 160   # resolucao do grid de probabilidades
THRESH_RES  = 96    # grid menor para a fronteira dinamica do slider

N_BALANCED  = 400
N_MAJ       = 700
N_MIN       = 100

_NEXT_MODE = {"linear": "quadratico", "quadratico": "impossivel", "impossivel": "linear"}
_BTN_LABEL = {"linear": "→ Quadrático", "quadratico": "→ Impossível", "impossivel": "→ Linear"}
_MODE_DESC = {
    "linear":     "Linear  [x₁, x₂]",
    "quadratico": "Quadrático  [x₁, x₂, x₁², x₂², x₁x₂]",
    "impossivel": "Impossível  (anéis — logística sempre falha)",
}

# ── dados ────────────────────────────────────────────────────────────────────

def _sample_annulus(rng: np.random.Generator, n: int,
                    r_min: float, r_max: float) -> np.ndarray:
    pts: list[np.ndarray] = []
    while sum(len(p) for p in pts) < n:
        xy = rng.uniform(-r_max, r_max, (max(n * 6, 600), 2))
        r  = np.hypot(xy[:, 0], xy[:, 1])
        pts.append(xy[(r >= r_min) & (r <= r_max)])
    return np.vstack(pts)[:n]


def generate(mode: str, imbalanced: bool) -> tuple[np.ndarray, np.ndarray]:
    rng    = np.random.default_rng(SEED)
    n0, n1 = (N_MAJ, N_MIN) if imbalanced else (N_BALANCED, N_BALANCED)

    if mode == "linear":
        COV = np.array([[0.8, 0.0], [0.0, 0.8]])
        X0  = rng.multivariate_normal([-1.5, 0.0], COV, n0)
        X1  = rng.multivariate_normal([ 1.5, 0.0], COV, n1)
    elif mode == "quadratico":
        X0  = rng.multivariate_normal([0.0, 0.0],
                                       [[2.2, 0.4], [0.4, 1.5]], n0)
        X1  = rng.multivariate_normal([1.5, 0.5],
                                       [[0.35, 0.25], [0.25, 0.40]], n1)
    else:  # impossivel: classe 0 = disco + anel externo; classe 1 = anel do meio
        h   = n0 // 2
        X0a = _sample_annulus(rng, h,     0.00, 0.85)
        X0b = _sample_annulus(rng, n0-h,  2.35, 3.60)
        X0  = np.vstack([X0a, X0b])
        X1  = _sample_annulus(rng, n1,    0.95, 2.25)

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n0), np.ones(n1)])
    return X, y

# ── modelo ───────────────────────────────────────────────────────────────────

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def _features(X: np.ndarray, quad: bool) -> np.ndarray:
    x1, x2 = X[:, 0], X[:, 1]
    bias   = np.ones(len(X))
    if quad:
        return np.column_stack([x1, x2, x1**2, x2**2, x1*x2, bias])
    return np.column_stack([x1, x2, bias])


def fit_logistic(X: np.ndarray, y: np.ndarray, quad: bool,
                 lr: float = 0.3, n_iter: int = 1500) -> np.ndarray:
    Xa = _features(X, quad)
    w  = np.zeros(Xa.shape[1])
    for _ in range(n_iter):
        w -= lr * Xa.T @ (sigmoid(Xa @ w) - y) / len(y)
    return w


def get_scores(X: np.ndarray, w: np.ndarray, quad: bool) -> np.ndarray:
    return sigmoid(_features(X, quad) @ w)

# ── métricas ─────────────────────────────────────────────────────────────────

def roc_curve(y: np.ndarray, scores: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
    ths         = np.linspace(0.0, 1.0, 300)
    fprs, tprs  = np.empty(300), np.empty(300)
    pos, neg    = y.sum(), (1 - y).sum()
    for i, t in enumerate(ths):
        pred    = scores >= t
        tprs[i] = (pred & (y == 1)).sum() / pos if pos else 0.0
        fprs[i] = (pred & (y == 0)).sum() / neg if neg else 0.0
    return fprs, tprs


def metrics_at(y: np.ndarray, scores: np.ndarray,
               thresh: float) -> dict[str, float]:
    pred = (scores >= thresh).astype(int)
    tp   = ((pred == 1) & (y == 1)).sum()
    fp   = ((pred == 1) & (y == 0)).sum()
    fn   = ((pred == 0) & (y == 1)).sum()
    tn   = ((pred == 0) & (y == 0)).sum()
    acc  = (tp + tn) / len(y)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"Acuracia": acc, "Precisao": prec, "Recall": rec, "F1": f1}

# ── app ───────────────────────────────────────────────────────────────────────

class LogisticSim:
    def __init__(self) -> None:
        self.mode       = "linear"
        self.imbalanced = False
        self._thresh_cs: object = None
        self._current_thresh: float | None = None

        self.fig = plt.figure(figsize=(15, 6))
        self.fig.canvas.manager.set_window_title(
            "Regressao Logistica — threshold e metricas")
        self.fig.subplots_adjust(
            left=0.06, right=0.97, top=0.86, bottom=0.20, wspace=0.38)

        self.ax_scatter = self.fig.add_subplot(1, 3, 1)
        self.ax_metrics = self.fig.add_subplot(1, 3, 2)
        self.ax_roc     = self.fig.add_subplot(1, 3, 3)

        self.sl_thresh = Slider(
            self.fig.add_axes([0.06, 0.07, 0.50, 0.04]),
            "threshold", 0.01, 0.99, valinit=INIT_THRESH, valstep=0.01,
        )
        self.btn_mode = Button(
            self.fig.add_axes([0.60, 0.055, 0.16, 0.06]),
            "→ Quadrático", color="#e8f0ff", hovercolor="#c0d0ff",
        )
        self.btn_balance = Button(
            self.fig.add_axes([0.79, 0.055, 0.18, 0.06]),
            "→ Desbalanceado", color="#e8f4e8", hovercolor="#c8e8c8",
        )
        self.title = self.fig.suptitle("", fontsize=12, fontweight="bold")

        self._rebuild()

        self.sl_thresh.on_changed(lambda v: self._on_thresh(float(v)))
        self.btn_mode.on_clicked(self._on_mode)
        self.btn_balance.on_clicked(self._on_balance)

    # ── dataset / model ──────────────────────────────────────────────────────

    def _load_data(self) -> None:
        quad = (self.mode != "linear")
        X, y = generate(self.mode, self.imbalanced)
        self.X, self.y = X, y
        self.quad = quad
        self.w      = fit_logistic(X, y, quad)
        self.scores = get_scores(X, self.w, quad)
        self.fprs, self.tprs = roc_curve(y, self.scores)
        self.auc = float(np.trapezoid(self.tprs[::-1], self.fprs[::-1]))

        pad = 0.7
        self._xlim = (X[:, 0].min() - pad, X[:, 0].max() + pad)
        self._ylim = (X[:, 1].min() - pad, X[:, 1].max() + pad)
        self.xx, self.yy, self.prob_map = self._make_prob_grid(GRID_RES)
        self.thresh_xx, self.thresh_yy, self.thresh_prob_map = (
            self._make_prob_grid(THRESH_RES)
        )

    def _make_prob_grid(
        self, resolution: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gx = np.linspace(*self._xlim, resolution)
        gy = np.linspace(*self._ylim, resolution)
        xx, yy = np.meshgrid(gx, gy)
        grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
        prob_map = get_scores(grid_pts, self.w, self.quad).reshape(xx.shape)
        return xx, yy, prob_map

    # ── full rebuild ─────────────────────────────────────────────────────────

    def _rebuild(self) -> None:
        self._load_data()
        self._draw_scatter_base()
        self._init_metrics_panel()
        self._draw_roc()
        thresh = float(self.sl_thresh.val)
        self._update_scatter_thresh(thresh)
        self._update_metrics(thresh)
        self._update_roc_marker(thresh)
        self._update_header()
        self.fig.canvas.draw_idle()

    def _update_header(self) -> None:
        bal = "Desbalanceado (7:1)" if self.imbalanced else "Balanceado (1:1)"
        self.title.set_text(f"{_MODE_DESC[self.mode]}  —  {bal}")
        self.btn_mode.label.set_text(_BTN_LABEL[self.mode])
        self.btn_balance.label.set_text(
            "→ Balanceado" if self.imbalanced else "→ Desbalanceado")

    # ── fast threshold update ────────────────────────────────────────────────

    def _on_thresh(self, val: float) -> None:
        self._update_scatter_thresh(val)
        self._update_metrics(val)
        self._update_roc_marker(val)
        self.fig.canvas.draw_idle()

    # ── scatter panel ────────────────────────────────────────────────────────

    def _draw_scatter_base(self) -> None:
        """Drawn once per dataset — static elements only."""
        ax = self.ax_scatter
        ax.clear()
        self._thresh_cs = None
        self._current_thresh = None

        X, y = self.X, self.y
        ax.imshow(self.prob_map,
                  extent=[*self._xlim, *self._ylim],
                  origin="lower", aspect="auto",
                  cmap="RdBu_r", alpha=0.22, vmin=0, vmax=1)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c="#2471a3", s=10,
                   alpha=0.55, label="classe 0")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c="#c0392b", s=10,
                   alpha=0.55, label="classe 1")
        ax.contour(self.xx, self.yy, self.prob_map,
                   levels=[0.5], colors=["#444"], linestyles=["--"],
                   linewidths=[1.5])
        ax.plot([], [], "--", color="#444", lw=1.5, label="fronteira t=0.50")
        ax.set_xlim(*self._xlim)
        ax.set_ylim(*self._ylim)
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.15)

    def _clear_threshold_contour(self) -> None:
        contour = self._thresh_cs
        if contour is None:
            return

        self._thresh_cs = None

        collections = getattr(contour, "collections", None)
        if collections is not None:
            for artist in list(collections):
                try:
                    artist.remove()
                except (ValueError, AttributeError):
                    pass

        try:
            contour.remove()
        except (ValueError, AttributeError):
            pass

    def _update_scatter_thresh(self, thresh: float) -> None:
        """Only swap the threshold contour — no ax.clear()."""
        thresh = round(float(thresh), 2)
        if self._current_thresh == thresh:
            return

        ax = self.ax_scatter
        self._clear_threshold_contour()
        if 0.0 < thresh < 1.0:
            self._thresh_cs = ax.contour(
                self.thresh_xx, self.thresh_yy, self.thresh_prob_map,
                levels=[thresh], colors=["#e67e22"],
                linewidths=[2.8], zorder=4)
        self._current_thresh = thresh
        ax.set_title(f"Dados  —  fronteira  t = {thresh:.2f}")

    # ── metrics panel ────────────────────────────────────────────────────────

    def _init_metrics_panel(self) -> None:
        ax = self.ax_metrics
        ax.clear()
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("valor")
        ax.grid(axis="y", alpha=0.18)
        colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]
        names  = ["Acuracia", "Precisao", "Recall", "F1"]
        self._bars = ax.bar(names, [0.0] * 4, color=colors,
                            alpha=0.85, width=0.55)
        self._bar_texts = [
            ax.text(b.get_x() + b.get_width() / 2, 0.02, "0.000",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
            for b in self._bars
        ]
        n_pos, n_neg = int(self.y.sum()), int((1 - self.y).sum())
        ax.text(0.98, 0.98, f"pos={n_pos}  neg={n_neg}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="#555")

    def _update_metrics(self, thresh: float) -> None:
        m = metrics_at(self.y, self.scores, thresh)
        for bar, txt, v in zip(self._bars, self._bar_texts, m.values()):
            bar.set_height(v)
            txt.set_position((bar.get_x() + bar.get_width() / 2, v + 0.02))
            txt.set_text(f"{v:.3f}")
        self.ax_metrics.set_title(f"Metricas  (threshold = {thresh:.2f})")

    # ── ROC panel ────────────────────────────────────────────────────────────

    def _draw_roc(self) -> None:
        ax = self.ax_roc
        ax.clear()
        ax.plot(self.fprs, self.tprs, color="#7f4acb", linewidth=2.5,
                label=f"ROC  (AUC = {self.auc:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="#aaa", linewidth=1.2,
                label="aleatório")
        self.roc_dot, = ax.plot([], [], "o", color="#e67e22",
                                markersize=10, zorder=5)
        ax.set_xlabel("FPR  (1 − especificidade)")
        ax.set_ylabel("TPR  (recall / sensibilidade)")
        ax.set_title("Curva ROC")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.18)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def _update_roc_marker(self, thresh: float) -> None:
        idx = int(np.round(thresh * (len(self.fprs) - 1) / 0.99))
        idx = np.clip(idx, 0, len(self.fprs) - 1)
        self.roc_dot.set_data([self.fprs[idx]], [self.tprs[idx]])

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _on_mode(self, _: object) -> None:
        self.mode = _NEXT_MODE[self.mode]
        self._rebuild()

    def _on_balance(self, _: object) -> None:
        self.imbalanced = not self.imbalanced
        self._rebuild()


def main() -> None:
    LogisticSim()
    plt.show()


if __name__ == "__main__":
    main()
