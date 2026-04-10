from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np


DOMAIN_MIN = 0.0
DOMAIN_MAX = 1.0


@dataclass(frozen=True)
class Dataset:
    x: np.ndarray
    y: np.ndarray
    y_true: np.ndarray


@dataclass(frozen=True)
class BiasVarianceDecomposition:
    degrees: np.ndarray
    bias_squared: np.ndarray
    variance: np.ndarray
    irreducible_error: np.ndarray
    total_error: np.ndarray


def standardize_domain(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 2.0 * (x - DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN) - 1.0


def legendre_to_x_coefficients(legendre_coefficients: np.ndarray) -> np.ndarray:
    standardized_as_x = np.polynomial.Polynomial(
        [
            -(DOMAIN_MAX + DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN),
            2.0 / (DOMAIN_MAX - DOMAIN_MIN),
        ]
    )
    polynomial_in_standardized = np.polynomial.Legendre(legendre_coefficients).convert(
        kind=np.polynomial.Polynomial
    )
    polynomial_in_x = polynomial_in_standardized(standardized_as_x)
    return np.asarray(polynomial_in_x.coef, dtype=float)


def build_true_coefficients(degree: int, seed: int = 20260409) -> np.ndarray:
    rng = np.random.default_rng(seed + degree * 97)
    legendre_coefficients = np.zeros(degree + 1, dtype=float)

    # All Legendre components are significant so that bias² decreases
    # visibly at each degree added (not just at the true degree).
    # Magnitude decays geometrically so the function remains smooth.
    decay = 0.55
    for k in range(degree + 1):
        magnitude = rng.uniform(0.6, 1.0) * (decay ** (degree - k))
        legendre_coefficients[k] = rng.choice([-1.0, 1.0]) * magnitude

    coefficients = legendre_to_x_coefficients(legendre_coefficients)
    grid = np.linspace(DOMAIN_MIN, DOMAIN_MAX, 512)
    values = evaluate_polynomial(coefficients, grid)
    coefficients[0] -= np.mean(values)

    normalized_values = evaluate_polynomial(coefficients, grid)
    scale = rng.uniform(0.75, 0.90) / np.max(np.abs(normalized_values))
    coefficients *= scale
    coefficients[0] += rng.uniform(-0.04, 0.04)
    return coefficients


def evaluate_polynomial(coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    values = np.zeros_like(x, dtype=float)

    for coefficient in coefficients[::-1]:
        values = values * x + coefficient

    return values


def generate_dataset(
    true_coefficients: np.ndarray,
    sample_count: int,
    noise_sigma: float,
    rng: np.random.Generator,
) -> Dataset:
    x = np.sort(rng.uniform(DOMAIN_MIN, DOMAIN_MAX, size=sample_count))
    y_true = evaluate_polynomial(true_coefficients, x)
    noise = rng.normal(loc=0.0, scale=noise_sigma, size=sample_count)
    y = y_true + noise
    return Dataset(x=x, y=y, y_true=y_true)


def fit_polynomial_regression(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    ridge_lambda: float = 1e-3,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    standardized_x = standardize_domain(x)
    design = np.polynomial.legendre.legvander(standardized_x, degree)
    gram = design.T @ design
    rhs = design.T @ y
    gram += ridge_lambda * np.eye(degree + 1)
    legendre_coefficients = np.linalg.solve(gram, rhs)
    return legendre_to_x_coefficients(legendre_coefficients)


def mean_squared_error(x: np.ndarray, y: np.ndarray, coefficients: np.ndarray) -> float:
    predictions = evaluate_polynomial(coefficients, x)
    return float(np.mean((predictions - y) ** 2))


def make_evaluation_grid(grid_size: int = 240) -> np.ndarray:
    return np.linspace(DOMAIN_MIN, DOMAIN_MAX, grid_size)


def _compute_degree_metrics(
    degree: int,
    true_coefficients: np.ndarray,
    true_values: np.ndarray,
    grid: np.ndarray,
    sample_count: int,
    noise_sigma: float,
    simulation_count: int,
    seed: int,
) -> tuple[int, float, float]:
    predictions = np.empty((simulation_count, grid.size), dtype=float)

    for simulation_index in range(simulation_count):
        rng = np.random.default_rng(seed + degree * 1000 + simulation_index * 37)
        dataset = generate_dataset(
            true_coefficients=true_coefficients,
            sample_count=sample_count,
            noise_sigma=noise_sigma,
            rng=rng,
        )
        fitted = fit_polynomial_regression(dataset.x, dataset.y, degree=degree)
        predictions[simulation_index] = evaluate_polynomial(fitted, grid)

    mean_prediction = predictions.mean(axis=0)
    bias_squared = float(np.mean((mean_prediction - true_values) ** 2))
    variance = float(np.mean(np.var(predictions, axis=0)))
    return degree, bias_squared, variance


def compute_bias_variance_decomposition(
    true_coefficients: np.ndarray,
    max_degree: int,
    sample_count: int,
    noise_sigma: float,
    simulation_count: int = 90,
    evaluation_grid: np.ndarray | None = None,
    seed: int = 31415,
    n_jobs: int = 1,
) -> BiasVarianceDecomposition:
    grid = make_evaluation_grid(180) if evaluation_grid is None else evaluation_grid
    true_values = evaluate_polynomial(true_coefficients, grid)
    degrees = np.arange(max_degree + 1)
    bias_squared = np.zeros_like(degrees, dtype=float)
    variance = np.zeros_like(degrees, dtype=float)
    irreducible_error = np.full_like(degrees, noise_sigma**2, dtype=float)
    worker_count = max(1, min(int(n_jobs), len(degrees)))

    if worker_count == 1:
        results = [
            _compute_degree_metrics(
                degree=degree,
                true_coefficients=true_coefficients,
                true_values=true_values,
                grid=grid,
                sample_count=sample_count,
                noise_sigma=noise_sigma,
                simulation_count=simulation_count,
                seed=seed,
            )
            for degree in degrees
        ]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(
                executor.map(
                    _compute_degree_metrics,
                    degrees,
                    [true_coefficients] * len(degrees),
                    [true_values] * len(degrees),
                    [grid] * len(degrees),
                    [sample_count] * len(degrees),
                    [noise_sigma] * len(degrees),
                    [simulation_count] * len(degrees),
                    [seed] * len(degrees),
                )
            )

    for degree, degree_bias_squared, degree_variance in results:
        bias_squared[degree] = degree_bias_squared
        variance[degree] = degree_variance

    total_error = bias_squared + variance + irreducible_error

    return BiasVarianceDecomposition(
        degrees=degrees,
        bias_squared=bias_squared,
        variance=variance,
        irreducible_error=irreducible_error,
        total_error=total_error,
    )
