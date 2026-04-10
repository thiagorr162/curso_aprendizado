from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# --- configuracao ------------------------------------------------------
N_BALANCED   = 400   # pontos por classe no cenario balanceado
N_MAJ        = 900   # classe majoritaria no cenario desbalanceado
N_MIN        = 100   # classe minoritaria no cenario desbalanceado
MU0          = np.array([-1.2,  0.0])
MU1          = np.array([ 1.2,  0.5])
COV          = np.array([[1.2, 0.3], [0.3, 0.8]])
SEED         = 42
INIT_THRESH  = 0.5


# --- dados e modelo ----------------------------------------------------

def generate(imbalanced: bool) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    n0, n1 = (N_MAJ, N_MIN) if imbalanced else (N_BALANCED, N_BALANCED)
    X0 = rng.multivariate_normal(MU0, COV, n0)
    X1 = rng.multivariate_normal(MU1, COV, n1)
    X  = np.vstack([X0, X1])
    y  = np.hstack([np.zeros(n0), np.ones(n1)])
    return X, y


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def fit_logistic(X: np.ndarray, y: np.ndarray, lr: float = 0.5, n_iter: int = 500) -> np.ndarray:
    Xa = np.column_stack([X, np.ones(len(X))])
    w  = np.zeros(3)
    for _ in range(n_iter):
        w -= lr * Xa.T @ (sigmoid(Xa @ w) - y) / len(y)
    return w


def decision_boundary_x2(w: np.ndarray, x1: np.ndarray, thresh: float) -> np.ndarray:
    """x2 da fronteira P(y=1|x)=thresh dado x1."""
    logit = np.log(thresh / (1.0 - thresh))
    return (logit - w[0] * x1 - w[2]) / w[1]


def roc_curve(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.linspace(0.0, 1.0, 300)
    fprs, tprs = np.empty(len(thresholds)), np.empty(len(thresholds))
    pos, neg = y.sum(), (1 - y).sum()
    for i, t in enumerate(thresholds):
        pred = scores >= t
        tprs[i] = (pred & (y == 1)).sum() / pos if pos else 0.0
        fprs[i] = (pred & (y == 0)).sum() / neg if neg else 0.0
    return fprs, tprs


def metrics_at(y: np.ndarray, scores: np.ndarray, thresh: float) -> dict[str, float]:
    pred = (scores >= thresh).astype(int)
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    acc  = (tp + tn) / len(y)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"Acuracia": acc, "Precisao": prec, "Recall": rec, "F1": f1}


# --- app ---------------------------------------------------------------

class LogisticSim:
    def __init__(self) -> None:
        self.imbalanced = False
        X, y = generate(self.imbalanced)
        self.X, self.y = X, y
        w = fit_logistic(X, y)
        self.w = w
        Xa = np.column_stack([X, np.ones(len(X))])
        self.scores = sigmoid(Xa @ w)
        self.fprs, self.tprs = roc_curve(y, self.scores)
        self.auc = float(np.trapezoid(self.tprs[::-1], self.fprs[::-1]))

        self.fig = plt.figure(figsize=(15, 6))
        self.fig.canvas.manager.set_window_title("Regressao Logistica — threshold e metricas")
        self.fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.20, wspace=0.38)

        self.ax_scatter = self.fig.add_subplot(1, 3, 1)
        self.ax_metrics = self.fig.add_subplot(1, 3, 2)
        self.ax_roc     = self.fig.add_subplot(1, 3, 3)

        self.sl_thresh = Slider(
            self.fig.add_axes([0.15, 0.07, 0.55, 0.04]),
            "threshold", 0.01, 0.99, valinit=INIT_THRESH, valstep=0.01,
        )
        self.btn_balance = Button(
            self.fig.add_axes([0.78, 0.06, 0.16, 0.06]),
            "→ Desbalanceado", color="#e8f4e8", hovercolor="#c8e8c8",
        )

        self.title = self.fig.suptitle("", fontsize=14, fontweight="bold")
        self._update_title()
        self._draw_roc()          # painel fixo por dataset
        self._redraw(INIT_THRESH) # paineis que mudam com threshold

        self.sl_thresh.on_changed(self._on_thresh)
        self.btn_balance.on_clicked(self._on_toggle)

    def _update_title(self) -> None:
        label = "Desbalanceado (9:1)" if self.imbalanced else "Balanceado (1:1)"
        self.title.set_text(f"Regressao Logistica — {label}")
        self.btn_balance.label.set_text(
            "→ Balanceado" if self.imbalanced else "→ Desbalanceado"
        )

    def _switch_dataset(self) -> None:
        X, y = generate(self.imbalanced)
        self.X, self.y = X, y
        self.w = fit_logistic(X, y)
        Xa = np.column_stack([X, np.ones(len(X))])
        self.scores = sigmoid(Xa @ self.w)
        self.fprs, self.tprs = roc_curve(y, self.scores)
        self.auc = float(np.trapezoid(self.tprs[::-1], self.fprs[::-1]))

    # ------------------------------------------------------------------
    def _on_thresh(self, val: float) -> None:
        self._redraw(float(val))

    def _on_toggle(self, _: object) -> None:
        self.imbalanced = not self.imbalanced
        self._switch_dataset()
        self._update_title()
        self._draw_roc()
        self._redraw(float(self.sl_thresh.val))

    # ------------------------------------------------------------------
    def _redraw(self, thresh: float) -> None:
        self._draw_scatter(thresh)
        self._draw_metrics(thresh)
        self._update_roc_marker(thresh)
        self.fig.canvas.draw_idle()

    def _draw_scatter(self, thresh: float) -> None:
        ax = self.ax_scatter
        ax.clear()
        X, y = self.X, self.y
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c="#2980b9", s=12,
                   alpha=0.5, label="classe 0")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c="#c0392b", s=12,
                   alpha=0.5, label="classe 1")

        x1 = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300)
        if abs(self.w[1]) > 1e-6:
            x2_05  = decision_boundary_x2(self.w, x1, 0.5)
            x2_thr = decision_boundary_x2(self.w, x1, thresh)
            ax.plot(x1, x2_05,  "--", color="#888", linewidth=1.5,
                    label="fronteira (t=0.5)")
            ax.plot(x1, x2_thr, "-",  color="#e67e22", linewidth=2.5,
                    label=f"fronteira (t={thresh:.2f})")

        ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
        ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
        ax.set_title("Dados e fronteira de decisao")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.18)

    def _draw_metrics(self, thresh: float) -> None:
        ax = self.ax_metrics
        ax.clear()
        m = metrics_at(self.y, self.scores, thresh)
        names  = list(m.keys())
        values = list(m.values())
        colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]
        bars = ax.bar(names, values, color=colors, alpha=0.85, width=0.55)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.set_title(f"Metricas  (threshold = {thresh:.2f})")
        ax.set_ylabel("valor")
        ax.grid(axis="y", alpha=0.18)
        n_pos = int(self.y.sum())
        n_neg = int((1 - self.y).sum())
        ax.text(0.98, 0.98, f"pos={n_pos}  neg={n_neg}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#555")

    def _draw_roc(self) -> None:
        ax = self.ax_roc
        ax.clear()
        ax.plot(self.fprs, self.tprs, color="#7f4acb", linewidth=2.5,
                label=f"ROC  (AUC={self.auc:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="#aaa", linewidth=1.2, label="aleatório")
        self.roc_dot, = ax.plot([], [], "o", color="#e67e22", markersize=10, zorder=5)
        ax.set_xlabel("FPR  (1 - especificidade)")
        ax.set_ylabel("TPR  (recall / sensibilidade)")
        ax.set_title("Curva ROC")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.18)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def _update_roc_marker(self, thresh: float) -> None:
        # ponto na ROC correspondente ao threshold atual
        idx = int(np.round(thresh * (len(self.fprs) - 1) / 0.99))
        idx = np.clip(idx, 0, len(self.fprs) - 1)
        self.roc_dot.set_data([self.fprs[idx]], [self.tprs[idx]])


def main() -> None:
    LogisticSim()
    plt.show()


if __name__ == "__main__":
    main()
