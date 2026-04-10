from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from polynomial_tools import (
    BiasVarianceDecomposition,
    Dataset,
    build_true_coefficients,
    compute_bias_variance_decomposition,
    evaluate_polynomial,
    fit_polynomial_regression,
    make_evaluation_grid,
    mean_squared_error,
    generate_dataset,
)


class BiasVarianceExplorer:
    def __init__(self) -> None:
        self.initial_sample_count = 50
        self.initial_true_degree = 5
        self.initial_model_degree = 5
        self.initial_noise = 0.05
        self.initial_simulations = 1
        self.n_jobs = min(2, os.cpu_count() or 1)
        self.true_polynomial_seed = 20260409
        self.dataset_seed = 8123
        self.metrics_seed = 31415
        self.grid = make_evaluation_grid(240)
        self.decomposition_cache: dict[
            tuple[int, float, int, int, int], BiasVarianceDecomposition
        ] = {}
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.current_future: Future[BiasVarianceDecomposition] | None = None
        self.current_future_key: tuple[int, float, int, int, int] | None = None
        self.pending_request: (
            tuple[tuple[int, float, int, int, int], np.ndarray, float, int, int, int] | None
        ) = None
        self.requested_decomposition_key: tuple[int, float, int, int, int] | None = None

        self.figure = plt.figure(figsize=(14, 8), constrained_layout=False)
        self.figure.canvas.manager.set_window_title("Bias-Variance Explorer")
        grid_spec = self.figure.add_gridspec(
            nrows=7,
            ncols=12,
            left=0.05,
            right=0.98,
            top=0.92,
            bottom=0.08,
            hspace=0.9,
            wspace=0.9,
        )

        self.ax_fit = self.figure.add_subplot(grid_spec[:4, :6])
        self.ax_metrics = self.figure.add_subplot(grid_spec[:4, 6:])

        self.slider_true_degree = Slider(
            self.figure.add_subplot(grid_spec[4, 0:5]),
            "grau verdadeiro k",
            1,
            self.initial_sample_count,
            valinit=self.initial_true_degree,
            valstep=1,
            dragging=False,
        )
        self.slider_model_degree = Slider(
            self.figure.add_subplot(grid_spec[4, 6:11]),
            "grau da regressao",
            0,
            self.initial_sample_count,
            valinit=self.initial_model_degree,
            valstep=1,
        )
        self.slider_noise = Slider(
            self.figure.add_subplot(grid_spec[5, 0:3]),
            "ruido",
            0.01,
            0.10,
            valinit=self.initial_noise,
            valstep=0.01,
            dragging=False,
        )
        self.slider_sample_count = Slider(
            self.figure.add_subplot(grid_spec[5, 4:7]),
            "amostra",
            50,
            200,
            valinit=self.initial_sample_count,
            valstep=1,
            dragging=False,
        )
        self.slider_simulations = Slider(
            self.figure.add_subplot(grid_spec[5, 8:12]),
            "simulacoes",
            1,
            10,
            valinit=self.initial_simulations,
            valstep=1,
            dragging=False,
        )

        self.button_new_sample = Button(
            self.figure.add_subplot(grid_spec[6, 8:10]),
            "Nova amostra",
        )
        self.button_resimulate = Button(
            self.figure.add_subplot(grid_spec[6, 10:12]),
            "Recalcular",
        )

        self.figure.suptitle(
            "Trade-off entre vies e variancia em regressao polinomial",
            fontsize=18,
            fontweight="bold",
        )
        self.status_artist = self.figure.text(
            0.98,
            0.955,
            "",
            ha="right",
            va="center",
            fontsize=10,
            color="#6a4f3b",
        )

        self.update_true_degree_limit()
        self.update_model_degree_limit()
        self.true_coefficients = self.current_true_coefficients()
        self.current_dataset = self.make_dataset()
        self.decomposition = self.run_decomposition()

        self.connect_events()
        self.poll_timer = self.figure.canvas.new_timer(interval=120)
        self.poll_timer.add_callback(self.poll_background_job)
        self.poll_timer.start()
        self.redraw()

    def control_values(self) -> tuple[int, int, float, int, int]:
        true_degree = int(self.slider_true_degree.val)
        model_degree = int(self.slider_model_degree.val)
        noise_sigma = float(self.slider_noise.val)
        sample_count = int(self.slider_sample_count.val)
        simulation_count = int(self.slider_simulations.val)
        return true_degree, model_degree, noise_sigma, sample_count, simulation_count

    def current_true_coefficients(self) -> np.ndarray:
        true_degree, _, _, _, _ = self.control_values()
        return build_true_coefficients(true_degree, seed=self.true_polynomial_seed)

    def current_max_true_degree(self) -> int:
        return int(self.slider_sample_count.val)

    def update_true_degree_limit(self) -> None:
        max_true_degree = self.current_max_true_degree()
        self.slider_true_degree.valmax = max_true_degree
        self.slider_true_degree.ax.set_xlim(self.slider_true_degree.valmin, max_true_degree)

        current_true_degree = int(self.slider_true_degree.val)
        if current_true_degree > max_true_degree:
            self.slider_true_degree.set_val(max_true_degree)

    def current_max_degree(self) -> int:
        return max(0, int(self.slider_sample_count.val) - 1)

    def update_model_degree_limit(self) -> None:
        max_degree = self.current_max_degree()
        self.slider_model_degree.valmax = max_degree
        self.slider_model_degree.ax.set_xlim(self.slider_model_degree.valmin, max_degree)

        current_degree = int(self.slider_model_degree.val)
        if current_degree > max_degree:
            self.slider_model_degree.set_val(max_degree)

    def make_dataset(self) -> Dataset:
        _, _, noise_sigma, sample_count, _ = self.control_values()
        rng = np.random.default_rng(self.dataset_seed)
        return generate_dataset(
            true_coefficients=self.true_coefficients,
            sample_count=sample_count,
            noise_sigma=noise_sigma,
            rng=rng,
        )

    def run_decomposition(self) -> BiasVarianceDecomposition:
        true_degree, _, noise_sigma, sample_count, simulation_count = self.control_values()
        cache_key = (
            true_degree,
            round(noise_sigma, 2),
            sample_count,
            simulation_count,
            self.metrics_seed,
        )
        self.requested_decomposition_key = cache_key

        if cache_key not in self.decomposition_cache:
            self.decomposition_cache[cache_key] = self.compute_decomposition(
                true_coefficients=self.true_coefficients,
                noise_sigma=noise_sigma,
                sample_count=sample_count,
                simulation_count=simulation_count,
                metrics_seed=self.metrics_seed,
            )

        return self.decomposition_cache[cache_key]

    def connect_events(self) -> None:
        self.slider_true_degree.on_changed(self.on_structure_change)
        self.slider_noise.on_changed(self.on_structure_change)
        self.slider_sample_count.on_changed(self.on_structure_change)
        self.slider_simulations.on_changed(self.on_structure_change)
        self.slider_model_degree.on_changed(self.on_model_degree_change)
        self.button_new_sample.on_clicked(self.on_new_sample)
        self.button_resimulate.on_clicked(self.on_resimulate)
        self.figure.canvas.mpl_connect("close_event", self.on_close)

    def decomposition_key(self) -> tuple[int, float, int, int, int]:
        true_degree, _, noise_sigma, sample_count, simulation_count = self.control_values()
        return (
            true_degree,
            round(noise_sigma, 2),
            sample_count,
            simulation_count,
            self.metrics_seed,
        )

    def compute_decomposition(
        self,
        true_coefficients: np.ndarray,
        noise_sigma: float,
        sample_count: int,
        simulation_count: int,
        metrics_seed: int,
    ) -> BiasVarianceDecomposition:
        return compute_bias_variance_decomposition(
            true_coefficients=true_coefficients,
            max_degree=sample_count - 1,
            sample_count=sample_count,
            noise_sigma=noise_sigma,
            simulation_count=simulation_count,
            evaluation_grid=make_evaluation_grid(180),
            seed=metrics_seed,
            n_jobs=self.n_jobs,
        )

    def trim_cache(self, max_entries: int = 16) -> None:
        while len(self.decomposition_cache) > max_entries:
            oldest_key = next(iter(self.decomposition_cache))
            del self.decomposition_cache[oldest_key]

    def set_status(self, message: str) -> None:
        self.status_artist.set_text(message)

    def request_decomposition(self) -> None:
        _, _, noise_sigma, sample_count, simulation_count = self.control_values()
        cache_key = self.decomposition_key()
        self.requested_decomposition_key = cache_key

        if cache_key in self.decomposition_cache:
            self.decomposition = self.decomposition_cache[cache_key]
            if self.current_future is None:
                self.set_status("")
            return

        self.pending_request = (
            cache_key,
            self.true_coefficients.copy(),
            noise_sigma,
            sample_count,
            simulation_count,
            self.metrics_seed,
        )

        if self.current_future is None:
            self.start_pending_request()
        else:
            self.set_status("Calculando... atualizacao pendente")

    def start_pending_request(self) -> None:
        if self.pending_request is None:
            return

        (
            cache_key,
            true_coefficients,
            noise_sigma,
            sample_count,
            simulation_count,
            metrics_seed,
        ) = self.pending_request
        self.pending_request = None
        self.current_future_key = cache_key
        self.current_future = self.executor.submit(
            self.compute_decomposition,
            true_coefficients,
            noise_sigma,
            sample_count,
            simulation_count,
            metrics_seed,
        )
        self.set_status(
            f"Calculando decomposicao ({simulation_count} simulacoes, {self.n_jobs} jobs)..."
        )

    def poll_background_job(self) -> None:
        if self.current_future is None or not self.current_future.done():
            return

        future = self.current_future
        cache_key = self.current_future_key
        self.current_future = None
        self.current_future_key = None

        try:
            result = future.result()
        except Exception as exc:
            self.set_status(f"Erro no recalculo: {exc}")
            self.figure.canvas.draw_idle()
            if self.pending_request is not None:
                self.start_pending_request()
            return

        assert cache_key is not None
        self.decomposition_cache[cache_key] = result
        self.trim_cache()

        if cache_key == self.requested_decomposition_key:
            self.decomposition = result
            self.set_status("")
            self.redraw()

        if self.pending_request is not None:
            self.start_pending_request()
            self.figure.canvas.draw_idle()

    def on_close(self, _event: object) -> None:
        self.poll_timer.stop()
        self.executor.shutdown(wait=False, cancel_futures=True)

    def on_structure_change(self, _value: float) -> None:
        self.update_true_degree_limit()
        self.update_model_degree_limit()
        self.true_coefficients = self.current_true_coefficients()
        self.current_dataset = self.make_dataset()
        self.request_decomposition()
        self.redraw()

    def on_model_degree_change(self, _value: float) -> None:
        self.redraw()

    def on_new_sample(self, _event: object) -> None:
        self.dataset_seed += 11
        self.current_dataset = self.make_dataset()
        self.redraw()

    def on_resimulate(self, _event: object) -> None:
        self.metrics_seed += 101
        self.request_decomposition()
        self.redraw()

    @staticmethod
    def smooth_series(values: np.ndarray, window_size: int = 5, log_space: bool = False) -> np.ndarray:
        if window_size <= 1 or values.size < window_size:
            return values.copy()

        working = np.log(np.maximum(values, 1e-9)) if log_space else values
        pad = window_size // 2
        padded = np.pad(working, (pad, pad), mode="edge")
        kernel = np.ones(window_size, dtype=float) / window_size
        smoothed = np.convolve(padded, kernel, mode="valid")
        return np.exp(smoothed) if log_space else smoothed

    def redraw(self) -> None:
        self.draw_fit_panel()
        self.draw_metrics_panel()
        self.figure.canvas.draw_idle()

    def draw_fit_panel(self) -> None:
        _, model_degree, _, _, _ = self.control_values()
        fitted_coefficients = fit_polynomial_regression(
            self.current_dataset.x,
            self.current_dataset.y,
            degree=model_degree,
        )
        truth = evaluate_polynomial(self.true_coefficients, self.grid)
        prediction = evaluate_polynomial(fitted_coefficients, self.grid)
        mse = mean_squared_error(
            self.current_dataset.x,
            self.current_dataset.y,
            fitted_coefficients,
        )

        self.ax_fit.clear()
        self.ax_fit.scatter(
            self.current_dataset.x,
            self.current_dataset.y,
            color="#40322a",
            s=32,
            label="dados observados",
            zorder=3,
        )
        self.ax_fit.plot(
            self.grid,
            truth,
            linestyle="--",
            linewidth=2.6,
            color="#2a7f9e",
            label="funcao verdadeira",
        )
        self.ax_fit.plot(
            self.grid,
            prediction,
            linewidth=2.8,
            color="#c55f3b",
            label="funcao predita",
        )
        self.ax_fit.set_title("Amostra observada e funcao ajustada")
        self.ax_fit.set_xlabel("x")
        self.ax_fit.set_ylabel("y")
        self.ax_fit.grid(alpha=0.18)
        self.ax_fit.legend(loc="upper left")
        self.ax_fit.text(
            0.02,
            0.03,
            f"MSE na amostra = {mse:.3f}",
            transform=self.ax_fit.transAxes,
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f4e8d8", "edgecolor": "none"},
        )

    def draw_metrics_panel(self) -> None:
        _, model_degree, _, sample_count, simulation_count = self.control_values()
        metrics = self.decomposition
        selected_index = model_degree
        max_degree = int(metrics.degrees[-1])
        axis_max_degree = sample_count
        best_degree = int(np.argmin(metrics.total_error))
        degrees = metrics.degrees

        # left arm peak: highest total error to the left of best_degree
        left_peak = float(np.max(metrics.total_error[:best_degree])) if best_degree > 0 else float(metrics.total_error[0])

        # right bound: last degree where total_error <= left_peak (so both arms match in height)
        right_ok = np.where((degrees > best_degree) & (metrics.total_error <= left_peak))[0]
        right_bound = int(degrees[right_ok[-1]]) if len(right_ok) > 0 else min(max_degree, best_degree + 10)

        # left bound: first degree (from 0) where total_error <= left_peak * 1.1
        left_ok = np.where((degrees <= best_degree) & (metrics.total_error <= left_peak * 1.1))[0]
        left_bound = int(degrees[left_ok[0]]) if len(left_ok) > 0 else 0

        margin = max(1, (right_bound - left_bound) // 6)
        visible_x_min = max(0, left_bound - margin)
        visible_x_max = min(max_degree, right_bound + margin)
        if visible_x_max - visible_x_min < 6:
            visible_x_max = min(max_degree, visible_x_min + 6)
        tick_step = max(1, int(np.ceil((visible_x_max - visible_x_min) / 8)))
        tick_values = list(range(visible_x_min, visible_x_max + 1, tick_step))
        if tick_values[-1] != visible_x_max:
            tick_values.append(visible_x_max)
        eps = 1e-6
        visible_mask = (metrics.degrees >= visible_x_min) & (metrics.degrees <= visible_x_max)
        visible_total = np.maximum(metrics.total_error[visible_mask], eps)
        use_log_scale = (
            float(np.max(visible_total)) / float(np.min(visible_total)) > 20.0
        )
        bias_for_plot = np.maximum(metrics.bias_squared, eps)
        variance_for_plot = np.maximum(metrics.variance, eps)
        irreducible_for_plot = np.maximum(metrics.irreducible_error, eps)
        total_for_plot = np.maximum(metrics.total_error, eps)
        total_smoothed = self.smooth_series(total_for_plot, window_size=5, log_space=use_log_scale)

        # y limits based only on visible window
        all_visible = np.concatenate([
            bias_for_plot[visible_mask],
            variance_for_plot[visible_mask],
            irreducible_for_plot[visible_mask],
            total_for_plot[visible_mask],
        ])
        y_min = float(np.min(all_visible))
        y_max = float(np.max(all_visible))
        if use_log_scale:
            y_pad = (np.log10(y_max) - np.log10(y_min)) * 0.15
            y_lim = (10 ** (np.log10(y_min) - y_pad), 10 ** (np.log10(y_max) + y_pad))
        else:
            y_pad = (y_max - y_min) * 0.15
            y_lim = (max(0.0, y_min - y_pad), y_max + y_pad)

        self.ax_metrics.clear()
        self.ax_metrics.plot(
            metrics.degrees,
            bias_for_plot,
            linewidth=2.5,
            color="#2a7f9e",
            label="vies ao quadrado",
        )
        self.ax_metrics.plot(
            metrics.degrees,
            variance_for_plot,
            linewidth=2.5,
            color="#2b8b67",
            label="variancia",
        )
        self.ax_metrics.plot(
            metrics.degrees,
            irreducible_for_plot,
            linewidth=2.5,
            color="#d69a1e",
            label="erro irredutivel",
        )
        self.ax_metrics.plot(
            metrics.degrees,
            total_for_plot,
            linewidth=1.2,
            color="#7f4acb",
            alpha=0.35,
            label="_nolegend_",
        )
        self.ax_metrics.scatter(
            metrics.degrees,
            total_for_plot,
            color="#7f4acb",
            s=16,
            alpha=0.35,
            zorder=3,
            label="_nolegend_",
        )
        self.ax_metrics.plot(
            metrics.degrees,
            total_smoothed,
            linewidth=3.2,
            color="#7f4acb",
            label="soma total",
        )
        self.ax_metrics.axvline(
            x=model_degree,
            color="#c55f3b",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
        )
        self.ax_metrics.scatter(
            [model_degree],
            [max(float(metrics.total_error[selected_index]), eps)],
            color="#c55f3b",
            s=50,
            zorder=4,
        )
        self.ax_metrics.set_title("Decomposicao do erro por grau do modelo")
        self.ax_metrics.set_xlabel("grau da regressao")
        self.ax_metrics.set_ylabel("erro esperado")
        self.ax_metrics.set_xlim(visible_x_min, visible_x_max)
        self.ax_metrics.set_ylim(*y_lim)
        self.ax_metrics.set_xticks(tick_values)
        self.ax_metrics.set_yscale("log" if use_log_scale else "linear")
        self.ax_metrics.grid(alpha=0.18)
        self.ax_metrics.legend(loc="upper left")
        self.ax_metrics.text(
            0.02,
            0.03,
            (
                f"grau {model_degree}: vies^2={metrics.bias_squared[selected_index]:.3f} | "
                f"variancia={metrics.variance[selected_index]:.3f} | "
                f"total={metrics.total_error[selected_index]:.3f} | "
                f"sim={simulation_count}"
            ),
            transform=self.ax_metrics.transAxes,
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#ede5fa", "edgecolor": "none"},
        )
        self.ax_metrics.text(
            0.98,
            0.92,
            (
                f"zoom: graus 1 a {visible_x_max} | melhor grau {best_degree} | "
                f"eixo y {'log' if use_log_scale else 'linear'}"
            ),
            transform=self.ax_metrics.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color="#6a4f3b",
        )


def main() -> None:
    BiasVarianceExplorer()
    plt.show()


if __name__ == "__main__":
    main()
