from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from .metrics import StaticSphericalMetric
from .policies import SolverOptions


def _safe_g(metric: StaticSphericalMetric, u: float, b: float) -> float:
    value = metric.G(float(u), float(b))
    if not math.isfinite(value):
        return -math.inf
    return value


def _scan_grid(
    u_start: float,
    u_stop: float,
    options: SolverOptions,
    *,
    allow_descending: bool,
    clip_start_at_zero: bool,
) -> np.ndarray | None:
    if allow_descending:
        if math.isclose(u_start, u_stop):
            return None
        low, high = sorted([u_start, u_stop])
        grid = np.linspace(low, high, options.max_brackets)
        if u_stop < u_start:
            grid = grid[::-1]
        return grid

    if u_stop <= u_start:
        return None
    return np.linspace(
        max(u_start, 0.0) if clip_start_at_zero else u_start,
        u_stop,
        options.max_brackets,
    )


def _brentq_turning_root(
    metric: StaticSphericalMetric,
    b: float,
    previous_u: float,
    current_u: float,
    options: SolverOptions,
) -> float:
    bracket = sorted([previous_u, current_u])
    return float(
        brentq(
            lambda u: metric.G(u, b),
            bracket[0],
            bracket[1],
            xtol=options.root_atol,
            rtol=options.root_rtol,
        )
    )


def _root_from_values(
    metric: StaticSphericalMetric,
    b: float,
    grid: np.ndarray,
    values: np.ndarray,
    options: SolverOptions,
) -> float | None:
    previous_u = float(grid[0])
    previous_g = float(values[0])
    for current_u_value, current_g_value in zip(grid[1:], values[1:]):
        current_u = float(current_u_value)
        current_g = float(current_g_value)
        if previous_g > 0.0 and current_g <= 0.0:
            return _brentq_turning_root(metric, b, previous_u, current_u, options)
        previous_u = current_u
        previous_g = current_g
    return None


def _tabulated_scan_supported(metric: StaticSphericalMetric, grid: np.ndarray) -> bool:
    from .numerical_metrics import TabulatedMetric

    if not isinstance(metric, TabulatedMetric):
        return False
    if np.any(~(grid > 0.0)):
        return False
    r = 1.0 / grid
    if np.any(~np.isfinite(r)):
        return False

    lo, hi = metric.valid_radial_domain()
    return bool(
        np.all(r > lo)
        and np.all(r < hi)
        and np.all(r >= metric._r_grid_min)
        and np.all(r <= metric._r_grid_max)
    )


def _first_turning_point_scalar_scan(
    metric: StaticSphericalMetric,
    b: float,
    u_start: float,
    u_stop: float,
    options: SolverOptions,
    *,
    allow_descending: bool = False,
    clip_start_at_zero: bool = False,
) -> float | None:
    grid = _scan_grid(
        u_start,
        u_stop,
        options,
        allow_descending=allow_descending,
        clip_start_at_zero=clip_start_at_zero,
    )
    if grid is None:
        return None

    previous_u = float(grid[0])
    previous_g = _safe_g(metric, previous_u, b)
    for current_u_value in grid[1:]:
        current_u = float(current_u_value)
        current_g = _safe_g(metric, current_u, b)
        if previous_g > 0.0 and current_g <= 0.0:
            return _brentq_turning_root(metric, b, previous_u, current_u, options)
        previous_u = current_u
        previous_g = current_g
    return None


def _first_turning_point_scan(
    metric: StaticSphericalMetric,
    b: float,
    u_start: float,
    u_stop: float,
    options: SolverOptions,
    *,
    allow_descending: bool = False,
    clip_start_at_zero: bool = False,
) -> float | None:
    grid = _scan_grid(
        u_start,
        u_stop,
        options,
        allow_descending=allow_descending,
        clip_start_at_zero=clip_start_at_zero,
    )
    if grid is None:
        return None

    if _tabulated_scan_supported(metric, grid):
        values = metric._G_array(grid, b)
        return _root_from_values(metric, b, grid, values, options)

    return _first_turning_point_scalar_scan(
        metric,
        b,
        u_start,
        u_stop,
        options,
        allow_descending=allow_descending,
        clip_start_at_zero=clip_start_at_zero,
    )
