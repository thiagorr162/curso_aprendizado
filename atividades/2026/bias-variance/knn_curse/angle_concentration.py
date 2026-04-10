from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib


if matplotlib.get_backend().lower() != "agg":
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


OUTFILE = Path(__file__).with_name("angle_concentration.png")
SEED = 20260409
N_ANGLE_PAIRS = 40_000
N_VOLUME_POINTS = 200_000
MAX_DIM = 30
INITIAL_DIM = 2
ANGLE_BINS = np.linspace(0.0, 180.0, 46)


@dataclass
class DimensionResult:
    dim: int
    angles_deg: np.ndarray
    inside_count: int
    volume_empirical: float
    mean_angle: float
    std_angle: float
    prob_80_100: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explora a concentracao de angulos e a perda de volume da esfera em alta dimensao."
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=MAX_DIM,
        help="maior dimensao disponivel no slider",
    )
    parser.add_argument(
        "--initial-dim",
        type=int,
        default=INITIAL_DIM,
        help="dimensao inicial mostrada ao abrir a figura",
    )
    parser.add_argument(
        "--angle-pairs",
        type=int,
        default=N_ANGLE_PAIRS,
        help="numero de pares de vetores usados no histograma",
    )
    parser.add_argument(
        "-N",
        "--volume-points",
        type=int,
        default=N_VOLUME_POINTS,
        help="numero de pontos uniformes no cubo para estimar a curva empirica de volume da esfera",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="semente aleatoria",
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="salva um snapshot PNG e nao abre janela interativa",
    )
    return parser.parse_args()


def sample_uniform_cube(dim: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=(n_samples, dim))


def sample_angles_in_degrees(dim: int, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    x = sample_uniform_cube(dim, n_pairs, rng)
    y = sample_uniform_cube(dim, n_pairs, rng)

    dot = np.sum(x * y, axis=1)
    norms = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    cos_theta = np.clip(dot / norms, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def empirical_sphere_volume_curve(
    max_dim: int,
    n_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    dims = np.arange(1, max_dim + 1, dtype=int)
    points = sample_uniform_cube(max_dim, n_points, rng)
    squared_norms = np.cumsum(points * points, axis=1)
    inside_counts = np.sum(squared_norms <= 1.0, axis=0).astype(int)
    cube_volumes = np.exp(dims * math.log(2.0))
    volume_estimates = cube_volumes * (inside_counts / n_points)
    return inside_counts, volume_estimates


def theoretical_sphere_volume(max_dim: int) -> np.ndarray:
    dims = np.arange(1, max_dim + 1, dtype=int)
    log_volume = np.array(
        [
            0.5 * dim * math.log(math.pi) - math.lgamma(0.5 * dim + 1.0)
            for dim in dims
        ],
        dtype=float,
    )
    return np.exp(log_volume)


class CubeCurseExplorer:
    def __init__(
        self,
        max_dim: int,
        initial_dim: int,
        n_angle_pairs: int,
        n_volume_points: int,
        seed: int,
    ) -> None:
        self.max_dim = max_dim
        self.current_dim = max(1, min(initial_dim, max_dim))
        self.n_angle_pairs = n_angle_pairs
        self.n_volume_points = n_volume_points
        self.seed = seed

        self.dim_grid = np.arange(1, self.max_dim + 1, dtype=int)
        self.theoretical_volume = theoretical_sphere_volume(self.max_dim)
        volume_rng = self._rng_for_curve()
        self.volume_counts_curve, self.empirical_volume_curve = empirical_sphere_volume_curve(
            self.max_dim,
            self.n_volume_points,
            volume_rng,
        )
        self.results: dict[int, DimensionResult] = {}

        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title("Curse of Dimensionality - cube")

        gs = self.fig.add_gridspec(
            2,
            2,
            left=0.07,
            right=0.97,
            top=0.90,
            bottom=0.16,
            hspace=0.28,
            wspace=0.24,
            width_ratios=[1.45, 1.0],
            height_ratios=[1.0, 0.72],
        )
        self.ax_hist = self.fig.add_subplot(gs[:, 0])
        self.ax_frac = self.fig.add_subplot(gs[0, 1])
        self.ax_text = self.fig.add_subplot(gs[1, 1])
        self.ax_text.axis("off")

        self.slider = Slider(
            self.fig.add_axes([0.16, 0.07, 0.68, 0.04]),
            "dimensao d",
            1,
            self.max_dim,
            valinit=self.current_dim,
            valstep=1,
        )
        self.slider.on_changed(self._on_slider_change)

        self.title = self.fig.suptitle("", fontsize=14, fontweight="bold")
        self._redraw()

    def _rng_for(self, dim: int, stream: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + dim * 1009 + stream * 7919)

    def _rng_for_curve(self) -> np.random.Generator:
        return np.random.default_rng(self.seed + 404_404)

    def _simulate_dimension(self, dim: int) -> DimensionResult:
        angle_rng = self._rng_for(dim, stream=1)
        angles = sample_angles_in_degrees(dim, self.n_angle_pairs, angle_rng)
        inside_count = int(self.volume_counts_curve[dim - 1])
        volume_emp = float(self.empirical_volume_curve[dim - 1])

        return DimensionResult(
            dim=dim,
            angles_deg=angles,
            inside_count=inside_count,
            volume_empirical=volume_emp,
            mean_angle=float(np.mean(angles)),
            std_angle=float(np.std(angles)),
            prob_80_100=float(np.mean((angles >= 80.0) & (angles <= 100.0))),
        )

    def _get_result(self, dim: int) -> DimensionResult:
        if dim not in self.results:
            self.results[dim] = self._simulate_dimension(dim)
        return self.results[dim]

    def _draw_histogram(self, result: DimensionResult) -> None:
        ax = self.ax_hist
        ax.clear()
        ax.hist(
            result.angles_deg,
            bins=ANGLE_BINS,
            density=True,
            color="#2a7f9e",
            edgecolor="white",
            linewidth=0.6,
            alpha=0.92,
        )
        ax.axvline(90.0, color="#c55f3b", linestyle="--", linewidth=2.0, label="90 graus")
        ax.set_xlim(0.0, 180.0)
        ax.set_xlabel("angulo entre vetores (graus)")
        ax.set_ylabel("densidade")
        ax.set_title(f"Histograma dos angulos para vetores uniformes em [-1, 1]^{result.dim}")
        ax.grid(alpha=0.20)
        ax.legend(fontsize=9, loc="upper left")

    def _draw_volume_panel(self, result: DimensionResult) -> None:
        ax = self.ax_frac
        ax.clear()
        ymax = float(np.max([np.max(self.theoretical_volume), np.max(self.empirical_volume_curve), 1.0]))

        ax.plot(
            self.dim_grid,
            self.theoretical_volume,
            color="#6a994e",
            linewidth=2.6,
            label="volume teorico da bola unitaria",
        )

        nonzero = self.empirical_volume_curve > 0.0
        if np.any(nonzero):
            ax.plot(
                self.dim_grid[nonzero],
                self.empirical_volume_curve[nonzero],
                color="#40322a",
                linewidth=1.8,
                marker="o",
                markersize=3.5,
                zorder=4,
                label=f"volume empirico (-N {self.n_volume_points})",
            )
        zero_mask = ~nonzero
        if np.any(zero_mask):
            ax.scatter(
                self.dim_grid[zero_mask],
                np.zeros(np.sum(zero_mask)),
                color="#40322a",
                marker="x",
                s=24,
                zorder=4,
                label="empirico = 0",
            )

        current_theory = self.theoretical_volume[result.dim - 1]
        current_emp = result.volume_empirical

        ax.axvline(result.dim, color="#c55f3b", linestyle="--", linewidth=1.8, alpha=0.85)
        ax.scatter([result.dim], [current_theory], color="#6a994e", s=80, zorder=5)
        ax.scatter([result.dim], [current_emp], color="#c55f3b", s=80, zorder=6)

        if result.volume_empirical == 0.0:
            ax.text(
                result.dim,
                ymax * 0.04,
                "0 no sample",
                color="#c55f3b",
                fontsize=9,
                ha="center",
            )

        ax.set_ylim(0.0, ymax * 1.12)
        ax.set_xlim(1, self.max_dim)
        ax.set_xlabel("dimensao d")
        ax.set_ylabel("volume estimado")
        ax.set_title("Volume da bola unitaria em funcao da dimensao")
        ax.grid(alpha=0.20, which="both")
        ax.legend(fontsize=8, loc="upper right")

        peak_dim = int(np.argmax(self.theoretical_volume) + 1)
        peak_volume = float(self.theoretical_volume[peak_dim - 1])
        ax.text(
            0.03,
            0.06,
            f"pico teorico em d = {peak_dim}  (V = {peak_volume:.3f})",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95},
        )

    def _draw_text_panel(self, result: DimensionResult) -> None:
        ax = self.ax_text
        ax.clear()
        ax.axis("off")

        theory = self.theoretical_volume[result.dim - 1]
        summary = "\n".join(
            [
                f"d = {result.dim}",
                "",
                f"angulo medio: {result.mean_angle:6.2f} graus",
                f"desvio padrao: {result.std_angle:6.2f} graus",
                f"P(80 <= angulo <= 100): {result.prob_80_100:6.3f}",
                "",
                f"dentro da esfera: {result.inside_count} / {self.n_volume_points}",
                f"volume empirico: {result.volume_empirical:8.5e}",
                f"volume teorico: {theory:8.5e}",
                "",
                "Leitura:",
                "os angulos vao se espremendo perto de 90 graus",
                "e o volume da bola atinge o pico em d = 5 e depois desaba.",
            ]
        )

        ax.text(
            0.02,
            0.96,
            summary,
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f7f4ea", "edgecolor": "#d8ceb6"},
        )

    def _redraw(self) -> None:
        result = self._get_result(self.current_dim)
        self._draw_histogram(result)
        self._draw_volume_panel(result)
        self._draw_text_panel(result)
        self.title.set_text(
            "Curse of dimensionality: angulos no cubo e volume da bola unitaria"
        )
        self.fig.canvas.draw_idle()

    def _on_slider_change(self, value: float) -> None:
        self.current_dim = int(value)
        self._redraw()


def main() -> None:
    args = parse_args()
    explorer = CubeCurseExplorer(
        max_dim=max(2, args.max_dim),
        initial_dim=args.initial_dim,
        n_angle_pairs=max(1000, args.angle_pairs),
        n_volume_points=max(1000, args.volume_points),
        seed=args.seed,
    )

    explorer.fig.savefig(OUTFILE, dpi=180)
    print(f"Snapshot salvo em: {OUTFILE}")
    print(
        "Abra a janela interativa e arraste o slider para ver a dimensao mudar."
        if not args.save_only and matplotlib.get_backend().lower() != "agg"
        else "Modo save-only: figura salva sem abrir janela."
    )

    if matplotlib.get_backend().lower() != "agg" and not args.save_only:
        plt.show()
    else:
        plt.close(explorer.fig)


if __name__ == "__main__":
    main()
