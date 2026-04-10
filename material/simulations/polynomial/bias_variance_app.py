from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from polynomial_tools import (
    build_true_coefficients,
    evaluate_polynomial,
    fit_polynomial_regression,
    generate_dataset,
    make_evaluation_grid,
)

NOISE      = 0.05
MAX_DEGREE = 15
N_DATASETS = 30
SEED_COEFF = 20260409
SEED_DATA  = 8123
RIDGE      = 1e-8

GRID = make_evaluation_grid(300)


def compute_test_errors(true_coeffs: np.ndarray, n: int, noise: float) -> np.ndarray:
    true_curve = evaluate_polynomial(true_coeffs, GRID)
    errors = np.zeros(MAX_DEGREE + 1)
    for i in range(N_DATASETS):
        rng = np.random.default_rng(SEED_DATA + i * 1000)
        ds  = generate_dataset(true_coeffs, n, noise, rng)
        for d in range(MAX_DEGREE + 1):
            c = fit_polynomial_regression(ds.x, ds.y, degree=d, ridge_lambda=RIDGE)
            errors[d] += np.mean((evaluate_polynomial(c, GRID) - true_curve) ** 2)
    return errors / N_DATASETS


class App:
    def __init__(self) -> None:
        self.n           = 30
        self.true_degree = 5
        self.model_degree = 5

        self.fig, (self.ax_fit, self.ax_err) = plt.subplots(1, 2, figsize=(13, 7))
        self.fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.22, wspace=0.35)

        # sliders
        self.sl_model  = Slider(self.fig.add_axes([0.10, 0.13, 0.80, 0.03]),
                                "grau do modelo", 0, MAX_DEGREE,
                                valinit=self.model_degree, valstep=1)
        self.sl_true   = Slider(self.fig.add_axes([0.10, 0.08, 0.22, 0.03]),
                                "grau real", 1, MAX_DEGREE,
                                valinit=self.true_degree, valstep=1, dragging=False)
        self.sl_n      = Slider(self.fig.add_axes([0.42, 0.08, 0.22, 0.03]),
                                "n (pontos)", 10, 100,
                                valinit=self.n, valstep=1, dragging=False)
        self.sl_noise  = Slider(self.fig.add_axes([0.74, 0.08, 0.22, 0.03]),
                                "ruido σ", 0.01, 0.10,
                                valinit=NOISE, valstep=0.01, dragging=False)

        # elementos do painel esquerdo (redesenhados ao mudar n / grau real)
        self.scatter_pts = None
        self.true_line   = None
        self.fit_line    = None

        # elementos do painel direito (redesenhados ao mudar n / grau real)
        self.err_line  = None
        self.best_line = None
        self.vline     = None
        self.dot       = None

        self._rebuild()

        self.sl_model.on_changed(lambda v: self._update_model(int(v)))
        self.sl_true.on_changed(lambda v: self._on_param_change())
        self.sl_n.on_changed(lambda v: self._on_param_change())
        self.sl_noise.on_changed(lambda v: self._on_param_change())

        plt.show()

    # ------------------------------------------------------------------
    def _rebuild(self) -> None:
        """Recomputa tudo quando n ou grau real mudam."""
        n            = int(self.sl_n.val)
        noise        = float(self.sl_noise.val)
        true_degree  = int(self.sl_true.val)
        model_degree = min(int(self.sl_model.val), MAX_DEGREE)

        self.true_coeffs = build_true_coefficients(true_degree, seed=SEED_COEFF)
        self.true_curve  = evaluate_polynomial(self.true_coeffs, GRID)
        self.test_errors = compute_test_errors(self.true_coeffs, n, noise)

        rng = np.random.default_rng(SEED_DATA)
        self.ds = generate_dataset(self.true_coeffs, n, noise, rng)

        self._draw_err_panel()
        self._draw_fit_panel(model_degree)

        self.fig.suptitle(
            f"Regressao polinomial — erro medio no grid  (media de {N_DATASETS} amostras)",
            fontsize=13, fontweight="bold",
        )
        self.fig.canvas.draw_idle()

    def _on_param_change(self) -> None:
        self._rebuild()

    def _update_model(self, degree: int) -> None:
        self._draw_fit_panel(degree)
        # atualiza marcador no painel de erro
        self.vline.set_xdata([degree, degree])
        self.dot.set_offsets([[degree, self.test_errors[degree]]])
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _draw_err_panel(self) -> None:
        ax = self.ax_err
        ax.clear()
        degrees = np.arange(MAX_DEGREE + 1)
        ax.plot(degrees, self.test_errors, color="#7f4acb", linewidth=2.8,
                marker="o", markersize=4, label="MSE grid vs f(x)")
        best = int(np.argmin(self.test_errors))
        self.best_line = ax.axvline(best, color="#aaa", linestyle=":",
                                    linewidth=1.5, label=f"melhor grau = {best}")
        model_degree = int(self.sl_model.val)
        self.vline = ax.axvline(model_degree, color="#c55f3b",
                                linestyle="--", linewidth=2, alpha=0.85)
        self.dot   = ax.scatter([model_degree], [self.test_errors[model_degree]],
                                color="#c55f3b", s=80, zorder=5)
        ax.set_xlabel("grau do modelo")
        ax.set_ylabel("MSE  vs  f(x) verdadeira  (log)")
        ax.set_title("Erro de teste no grid")
        ax.set_yscale("log")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.18, which="both")
        ax.set_xlim(0, MAX_DEGREE)

    def _draw_fit_panel(self, degree: int) -> None:
        ax = self.ax_fit
        ax.clear()
        c    = fit_polynomial_regression(self.ds.x, self.ds.y, degree=degree, ridge_lambda=RIDGE)
        pred = evaluate_polynomial(c, GRID)
        ax.scatter(self.ds.x, self.ds.y, color="#40322a", s=28, zorder=3,
                   alpha=0.75, label="dados treino")
        ax.plot(GRID, self.true_curve, "--", color="#2a7f9e",
                linewidth=2.5, label="f(x) verdadeira")
        ax.plot(GRID, pred, color="#c55f3b", linewidth=2.5, label="ajuste polinomial")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            f"Ajuste grau {degree}  —  MSE grid = {self.test_errors[degree]:.4f}"
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.18)
        ax.set_xlim(0, 1)


if __name__ == "__main__":
    App()
