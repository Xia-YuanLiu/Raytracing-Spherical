"""Numerical (tabulated) static spherical metric components.

This module exposes the data contract and interpolation objects required to
trace rays on numerically supplied ``r/A/B`` (or Li-DM-BH ``r/f/g/B/params``)
tables. The components are intentionally separated per plan decisions 3/4/8:

- :class:`LiDMBHMetadata` is a structured, immutable schema for the metadata
  carried alongside a Li-DM-BH ``.npz`` payload (or its ``raw_B is None``
  counterpart for the standard ``r/A/B`` format).
- :class:`CanonicalPayload` holds the canonical ``r, A, C, metadata`` arrays.
- :class:`ValidationOptions` exposes the hard-fail / soft-warn thresholds
  enumerated by plan decision 8.
- :class:`StaticDomainSelector` selects the active static patch from metadata
  (with optional explicit override) per plan decision 5.
- :class:`TabulatedMetric` realises the :class:`StaticSphericalMetric`
  protocol via PCHIP interpolation of ``A`` and ``C`` on the static-patch
  subarray with horizon endpoint anchors per plan decision 4.

The actual loader entry point ``load_metric_npz`` and ``ValidationGate`` live
in :mod:`spherical_raytracing.numerical_loader`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from .diagnostics import CriticalCurve


# Plan decision 6 horizon-match tolerance: used to decide whether a static-
# domain endpoint coincides with a metadata horizon (so it should be zero-
# anchored) and to classify event types in the solver.
_HORIZON_MATCH_REL_TOL = 1e-6
_HORIZON_MATCH_ABS_TOL = 1e-8


def _is_horizon_match(r_boundary: float, horizons: Sequence[float]) -> bool:
    """Return ``True`` when ``r_boundary`` matches any metadata horizon within
    the plan-decision-6 tolerance."""
    if not math.isfinite(r_boundary):
        return False
    for horizon in horizons:
        if not math.isfinite(horizon):
            continue
        if math.isclose(
            r_boundary,
            float(horizon),
            rel_tol=_HORIZON_MATCH_REL_TOL,
            abs_tol=_HORIZON_MATCH_ABS_TOL,
        ):
            return True
    return False


@dataclass(frozen=True)
class LiDMBHMetadata:
    """Structured Li-DM-BH metadata (plan decision 3).

    ``raw_B`` is populated only when the source is the Li-DM-BH ``r/f/g/B``
    format; in the standard ``r/A/B`` path ``raw_B`` is ``None`` because ``C``
    is computed directly from ``B`` and a separate consistency check is
    redundant.
    """

    source_format: str
    original_keys: tuple[str, ...]
    static_domains: tuple[tuple[float, float], ...]
    horizons: tuple[float, ...]
    photon_spheres: tuple[float, ...]
    conventions: Mapping[str, str]
    model_params: Mapping[str, Any]
    raw_B: np.ndarray | None = None

    @classmethod
    def from_params(
        cls,
        *,
        params: Mapping[str, Any] | None,
        source_format: str,
        original_keys: Sequence[str],
        raw_B: np.ndarray | None = None,
    ) -> "LiDMBHMetadata":
        """Build a ``LiDMBHMetadata`` from a raw ``params`` mapping.

        ``params`` mirrors the JSON-encoded ``params`` field of the Li-DM-BH
        ``.npz`` payload. ``None`` is accepted for the standard ``r/A/B``
        format where the upstream JSON metadata is absent.
        """
        params = dict(params) if params is not None else {}

        static_domains = tuple(
            (
                float(domain[0]),
                math.inf if domain[1] is None else float(domain[1]),
            )
            for domain in params.get("static_domains", [])
        )
        horizons = tuple(float(value) for value in params.get("horizons", []))
        photon_spheres = tuple(float(value) for value in params.get("photon_spheres", []))

        convention_keys = ("g_convention", "B_convention")
        conventions = {key: params[key] for key in convention_keys if key in params}

        reserved = set(convention_keys) | {"static_domains", "horizons", "photon_spheres"}
        model_params = {key: value for key, value in params.items() if key not in reserved}

        return cls(
            source_format=str(source_format),
            original_keys=tuple(original_keys),
            static_domains=static_domains,
            horizons=horizons,
            photon_spheres=photon_spheres,
            conventions=MappingProxyType(conventions),
            model_params=MappingProxyType(model_params),
            raw_B=np.asarray(raw_B, dtype=float) if raw_B is not None else None,
        )


@dataclass(frozen=True)
class CanonicalPayload:
    """Canonical ``r, A, C, metadata`` payload.

    ``C = 1/B`` is the primary interpolation target; ``B`` is reconstructed as
    ``1/C`` on demand. ``metadata.raw_B`` carries the original ``g_rr`` array
    only when the source format is Li-DM-BH (plan decision 3).
    """

    r: np.ndarray
    A: np.ndarray
    C: np.ndarray
    metadata: LiDMBHMetadata


@dataclass(frozen=True)
class ValidationOptions:
    """Hard-fail / soft-warn thresholds for the numerical metric pipeline.

    The defaults follow plan decision 8 directly; override them per-call from
    ``load_metric_npz`` when sensitivity tuning is needed (e.g. for regression
    sweeps).
    """

    bc_warn_lower: float = 1e-12
    bc_warn_upper: float = 1e-8
    horizon_residual_warn_lower: float = 1e-8
    horizon_residual_warn_upper: float = 1e-4
    photon_sphere_residual_warn: float = 1e-6
    c_tol_relative: float = 1e-12


@dataclass(frozen=True)
class StaticDomainSelector:
    """Static-domain selection helper (plan decision 5)."""

    domain_index: int = -1
    radial_bounds: tuple[float, float] | None = None

    def choose(self, payload: CanonicalPayload) -> tuple[float, float]:
        """Pick the active static domain ``(r_lo, r_hi)`` for ``payload``."""
        if self.radial_bounds is not None:
            r_lo, r_hi = self.radial_bounds
            r_lo_f = float(r_lo)
            r_hi_f = math.inf if r_hi is None or not math.isfinite(r_hi) else float(r_hi)
            if r_hi_f <= r_lo_f:
                raise ValueError(
                    f"radial_bounds must satisfy r_hi > r_lo (got r_lo={r_lo_f}, r_hi={r_hi_f})"
                )
            self._check_overlap(payload, r_lo_f, r_hi_f)
            return r_lo_f, r_hi_f

        domains = payload.metadata.static_domains
        if domains:
            try:
                r_lo, r_hi = domains[self.domain_index]
            except IndexError as exc:
                raise ValueError(
                    f"domain_index={self.domain_index} is out of range for "
                    f"{len(domains)} static domain(s)"
                ) from exc
            self._check_overlap(payload, r_lo, r_hi)
            return float(r_lo), float(r_hi)

        # Fallback inference from the table itself (plan decision 5).
        return _fallback_static_domain(payload)

    @staticmethod
    def _check_overlap(payload: CanonicalPayload, r_lo: float, r_hi: float) -> None:
        r_grid_min = float(np.min(payload.r))
        r_grid_max = float(np.max(payload.r))
        upper_grid = r_grid_max
        upper_domain = upper_grid if not math.isfinite(r_hi) else r_hi
        if upper_domain <= r_lo or r_lo >= r_grid_max or upper_domain <= r_grid_min:
            raise ValueError(
                "selected static domain does not overlap the tabulated support "
                f"(domain=({r_lo}, {r_hi}), grid=({r_grid_min}, {r_grid_max}))"
            )


def _fallback_static_domain(payload: CanonicalPayload) -> tuple[float, float]:
    """Largest contiguous ``A>0 & C>0`` interval of ``payload.r``."""
    positive = (payload.A > 0.0) & (payload.C > 0.0)
    if not np.any(positive):
        raise ValueError("no positive (A>0, C>0) contiguous interval found in payload")
    # Identify the longest True run.
    best_lo: int | None = None
    best_hi: int | None = None
    best_len = 0
    cur_lo: int | None = None
    for idx, ok in enumerate(positive):
        if ok and cur_lo is None:
            cur_lo = idx
        if (not ok or idx == len(positive) - 1) and cur_lo is not None:
            cur_hi = idx if not ok else idx + 1
            run_len = cur_hi - cur_lo
            if run_len > best_len:
                best_lo = cur_lo
                best_hi = cur_hi
                best_len = run_len
            cur_lo = None
    assert best_lo is not None and best_hi is not None
    return float(payload.r[best_lo]), float(payload.r[best_hi - 1])


class TabulatedMetric:
    """Numerical static spherical metric backed by a PCHIP-interpolated table.

    Implements the :class:`StaticSphericalMetric` protocol. Internally stores
    ``A(r)`` and ``C(r) = 1/B(r)`` because Li-DM-BH writes ``g = 1/B`` and
    interpolating ``B`` directly amplifies the visible-horizon divergence
    (plan decision 1).

    Anchoring strategy (plan decision 4):

    - A ``(r_boundary, 0.0)`` anchor is injected at a static-domain endpoint
      *only* when that endpoint matches a metadata horizon within plan
      decision 6 tolerance.
    - Non-horizon endpoints (e.g. a standard ``r/A/B`` table with no metadata
      horizons, or a user-supplied ``radial_bounds`` cropping inside a static
      patch) are *not* zero-anchored, because forcing ``A(r_boundary)=0``
      there would inject a fake horizon and distort values near the bound.
    - Infinite ``r_hi`` is never anchored (there is no finite boundary).

    All queries outside the static domain *or* beyond the tabulated grid
    support raise ``ValueError`` (plan decision 8 hard list); ``B`` /
    ``dB_dr`` queries where ``|C| < c_tol`` raise as well.

    When metadata ``horizons`` / ``photon_spheres`` are absent, the metric
    falls back to numerically inferring roots of ``A=0`` and
    ``r * A' - 2 * A = 0`` from the interpolated table. This keeps the
    documented standard ``r/A/B`` path workable: ``critical_curves()`` and
    ``b_crit`` remain available for callers without metadata.
    """

    def __init__(
        self,
        *,
        payload: CanonicalPayload,
        static_domain: tuple[float, float],
        region: str = "external",
        options: ValidationOptions | None = None,
    ) -> None:
        self.payload = payload
        self.region = region
        self._options = options if options is not None else ValidationOptions()

        r_lo, r_hi = float(static_domain[0]), float(static_domain[1])
        if r_hi <= r_lo:
            raise ValueError(
                f"static_domain must satisfy r_hi > r_lo (got {static_domain})"
            )
        self._domain = (r_lo, r_hi)

        r = payload.r
        A = payload.A
        C = payload.C

        meta = payload.metadata

        # Selective anchoring: only zero-anchor endpoints that match a
        # metadata horizon. This is what plan decision 4 ("horizon endpoint
        # anchors") literally requires; blindly anchoring at the support
        # bounds turns regular numerical support edges into fake horizons.
        anchor_lo = _is_horizon_match(r_lo, meta.horizons)
        anchor_hi = math.isfinite(r_hi) and _is_horizon_match(r_hi, meta.horizons)

        # When anchoring, the boundary sample comes from the synthetic anchor
        # (so the mask is strict to avoid duplicating ``r_lo`` / ``r_hi``).
        # When not anchoring, include grid samples on the boundary so the
        # PCHIP support extends to ``r_lo`` / ``r_hi`` even for tables whose
        # first / last sample sits exactly at the bound.
        mask_lo = (r > r_lo) if anchor_lo else (r >= r_lo)
        if math.isfinite(r_hi):
            mask_hi = (r < r_hi) if anchor_hi else (r <= r_hi)
        else:
            mask_hi = np.ones_like(r, dtype=bool)
        mask = mask_lo & mask_hi
        r_sub = np.asarray(r[mask], dtype=float)
        A_sub = np.asarray(A[mask], dtype=float)
        C_sub = np.asarray(C[mask], dtype=float)

        if len(r_sub) < 2:
            raise ValueError(
                "tabulated payload has fewer than 2 interior samples in the "
                f"selected static domain {self._domain}; cannot build PCHIP"
            )

        # Plan decision 8 (hard fail): A or C non-positive inside the selected
        # static domain. PCHIP is shape-preserving for monotone runs, but the
        # underlying sample must already be positive; relying on the endpoint
        # anchor alone is insufficient if the upstream table is bad.
        if np.any(A_sub <= 0.0):
            bad_idx = int(np.argmin(A_sub))
            raise ValueError(
                f"A_sub has non-positive sample inside static domain {self._domain}: "
                f"A[{bad_idx}]={A_sub[bad_idx]} at r={r_sub[bad_idx]}"
            )
        if np.any(C_sub <= 0.0):
            bad_idx = int(np.argmin(C_sub))
            raise ValueError(
                f"C_sub has non-positive sample inside static domain {self._domain}: "
                f"C[{bad_idx}]={C_sub[bad_idx]} at r={r_sub[bad_idx]}"
            )

        r_fit_parts: list[np.ndarray] = []
        A_fit_parts: list[np.ndarray] = []
        C_fit_parts: list[np.ndarray] = []
        if anchor_lo:
            r_fit_parts.append(np.array([r_lo]))
            A_fit_parts.append(np.array([0.0]))
            C_fit_parts.append(np.array([0.0]))
        r_fit_parts.append(r_sub)
        A_fit_parts.append(A_sub)
        C_fit_parts.append(C_sub)
        if anchor_hi:
            r_fit_parts.append(np.array([r_hi]))
            A_fit_parts.append(np.array([0.0]))
            C_fit_parts.append(np.array([0.0]))
        r_fit = np.concatenate(r_fit_parts)
        A_fit = np.concatenate(A_fit_parts)
        C_fit = np.concatenate(C_fit_parts)

        if np.any(np.diff(r_fit) <= 0):
            raise ValueError(
                "static-domain subarray is not strictly increasing after anchoring; "
                "check static_domain bounds vs payload.r"
            )

        self._A_pchip = PchipInterpolator(r_fit, A_fit, extrapolate=False)
        self._C_pchip = PchipInterpolator(r_fit, C_fit, extrapolate=False)
        self._dA_pchip = self._A_pchip.derivative()
        self._dC_pchip = self._C_pchip.derivative()
        c_scale = float(np.max(np.abs(C_sub))) if len(C_sub) else 1.0
        self._c_tol = self._options.c_tol_relative * max(c_scale, 1e-300)
        self._r_grid_max = float(r_fit.max())
        self._r_grid_min = float(r_fit.min())
        self._anchor_lo = bool(anchor_lo)
        self._anchor_hi = bool(anchor_hi)

        self._metadata_horizons = [float(h) for h in meta.horizons]
        self._metadata_photon_spheres = [float(p) for p in meta.photon_spheres]
        self._metadata_static_domains = [
            (float(lo), math.inf if not math.isfinite(hi) else float(hi))
            for lo, hi in meta.static_domains
        ] or [self._domain]
        # Caches for inferred roots; populated lazily on first access.
        self._inferred_horizons: list[float] | None = None
        self._inferred_photon_spheres: list[float] | None = None

    # ------------------------------------------------------------------
    # Core metric protocol
    # ------------------------------------------------------------------
    def _check_r(self, r: float) -> None:
        """Reject ``r`` outside the static domain *or* beyond the tabulated grid.

        Without the grid-support guard, PCHIP would silently return NaN
        outside ``[r_grid_min, r_grid_max]`` (PCHIP is built with
        ``extrapolate=False``) and pass NaN downstream into the solver,
        which then fails in confusing ways. Hard-failing here mirrors the
        guard already in :meth:`G`.

        The static-domain check uses an open interval ``(r_lo, r_hi)``
        (``r==r_lo`` is treated as "on the horizon" and rejected). The grid
        check uses a closed interval ``[r_grid_min, r_grid_max]`` because
        PCHIP supports those endpoints exactly.
        """
        lo, hi = self._domain
        if r <= lo or r >= hi:
            raise ValueError(
                f"r={r!r} outside static domain {self._domain} (TabulatedMetric)"
            )
        if r > self._r_grid_max:
            raise ValueError(
                f"r={r!r} above tabulated grid r_max={self._r_grid_max} "
                "(TabulatedMetric has no extrapolation support)"
            )
        if r < self._r_grid_min:
            raise ValueError(
                f"r={r!r} below tabulated grid r_min={self._r_grid_min} "
                "(TabulatedMetric has no extrapolation support)"
            )

    def A(self, r: float) -> float:
        self._check_r(r)
        return float(self._A_pchip(r))

    def B(self, r: float) -> float:
        self._check_r(r)
        c = float(self._C_pchip(r))
        if abs(c) < self._c_tol:
            raise ValueError(
                f"|C(r={r})|={abs(c):.3e} < c_tol={self._c_tol:.3e}; B is singular"
            )
        return 1.0 / c

    def dA_dr(self, r: float) -> float:
        self._check_r(r)
        return float(self._dA_pchip(r))

    def dB_dr(self, r: float) -> float:
        self._check_r(r)
        c = float(self._C_pchip(r))
        if abs(c) < self._c_tol:
            raise ValueError(
                f"|C(r={r})|={abs(c):.3e} < c_tol={self._c_tol:.3e}; dB_dr is singular"
            )
        return -float(self._dC_pchip(r)) / (c * c)

    def G(self, u: float, b: float) -> float:
        if b <= 0.0:
            raise ValueError("b must be positive")
        if u <= 0.0:
            raise ValueError(
                "TabulatedMetric.G requires u>0; r=1/u=inf has no tabulated support "
                "(use FiniteStaticObserver, not InfinityObserver)"
            )
        r = 1.0 / u
        if r > self._r_grid_max:
            raise ValueError(
                f"u={u} maps to r={r} above tabulated grid r_max={self._r_grid_max}"
            )
        self._check_r(r)
        a = float(self._A_pchip(r))
        c = float(self._C_pchip(r))
        return c * (1.0 / (a * b * b) - u * u)

    def _G_array(self, u_values: np.ndarray, b: float) -> np.ndarray:
        """Vectorized counterpart of :meth:`G` for private solver fast paths."""
        if not b > 0.0:
            raise ValueError("b must be positive")

        u = np.asarray(u_values, dtype=float)
        if np.any(~(u > 0.0)):
            raise ValueError(
                "TabulatedMetric.G requires u>0; r=1/u=inf has no tabulated support "
                "(use FiniteStaticObserver, not InfinityObserver)"
            )

        r = 1.0 / u
        if np.any(r > self._r_grid_max):
            raise ValueError(
                f"u_values map above tabulated grid r_max={self._r_grid_max}"
            )

        lo, hi = self._domain
        if np.any((r <= lo) | (r >= hi)):
            raise ValueError(
                f"r values outside static domain {self._domain} (TabulatedMetric)"
            )
        if np.any(r < self._r_grid_min):
            raise ValueError(
                f"u_values map below tabulated grid r_min={self._r_grid_min} "
                "(TabulatedMetric has no extrapolation support)"
            )

        a = np.asarray(self._A_pchip(r), dtype=float)
        c = np.asarray(self._C_pchip(r), dtype=float)
        if np.any(~np.isfinite(a)) or np.any(~np.isfinite(c)):
            raise ValueError("PCHIP interpolation returned non-finite values")
        return c * (1.0 / (a * b * b) - u * u)

    # ------------------------------------------------------------------
    # Roots: metadata first, otherwise infer from interpolated table
    # ------------------------------------------------------------------
    def horizons(self) -> list[float]:
        """Metadata horizons, or roots of ``A=0`` inferred from the table."""
        if self._metadata_horizons:
            return list(self._metadata_horizons)
        if self._inferred_horizons is None:
            self._inferred_horizons = self._infer_horizons()
        return list(self._inferred_horizons)

    def photon_spheres(self) -> list[float]:
        """Metadata ``r_ph`` values, or roots of ``rA'-2A=0`` from the table."""
        if self._metadata_photon_spheres:
            return list(self._metadata_photon_spheres)
        if self._inferred_photon_spheres is None:
            self._inferred_photon_spheres = self._infer_photon_spheres()
        return list(self._inferred_photon_spheres)

    def critical_curves(self) -> list[CriticalCurve]:
        curves: list[CriticalCurve] = []
        lo, hi = self._domain
        for index, r_ph in enumerate(self.photon_spheres()):
            if r_ph <= lo or r_ph >= hi:
                continue
            if r_ph <= self._r_grid_min or r_ph >= self._r_grid_max:
                continue
            a = float(self._A_pchip(r_ph))
            if a <= 0.0:
                continue
            curves.append(
                CriticalCurve(
                    id=f"{self.region}:photon_sphere:{index}",
                    r_ph=r_ph,
                    b_crit=r_ph / math.sqrt(a),
                    stability="unstable",
                    region=self.region,
                    reachable_from_observer=True,
                )
            )
        return curves

    def valid_radial_domain(self) -> tuple[float, float]:
        return self._domain

    def static_domains(self) -> list[tuple[float, float]]:
        return list(self._metadata_static_domains)

    # ------------------------------------------------------------------
    # Internal helpers for root inference
    # ------------------------------------------------------------------
    def _scan_brackets(self, samples: np.ndarray, f) -> list[tuple[float, float]]:
        """Walk ``samples`` collecting brackets where ``f`` changes sign."""
        brackets: list[tuple[float, float]] = []
        try:
            f_prev = float(f(samples[0]))
        except Exception:  # noqa: BLE001 - PCHIP boundary edge
            return brackets
        prev = float(samples[0])
        for r_next in samples[1:]:
            try:
                f_next = float(f(r_next))
            except Exception:  # noqa: BLE001
                prev = float(r_next)
                continue
            if (
                math.isfinite(f_prev)
                and math.isfinite(f_next)
                and f_prev * f_next < 0.0
            ):
                brackets.append((prev, float(r_next)))
            prev = float(r_next)
            f_prev = f_next
        return brackets

    def _refine_root(self, f, bracket: tuple[float, float]) -> float | None:
        try:
            return float(brentq(f, bracket[0], bracket[1], xtol=1e-12, rtol=1e-12))
        except Exception:  # noqa: BLE001 - degenerate bracket
            return None

    def _interior_sample_grid(self, num: int = 256) -> np.ndarray:
        """Dense scanning grid strictly inside the PCHIP support range."""
        eps = 1e-9
        lo = self._r_grid_min + eps
        hi = self._r_grid_max - eps
        if hi <= lo:
            return np.empty(0, dtype=float)
        # Mix the original sample grid with a uniform refinement so we don't
        # miss roots between widely-spaced samples (e.g. nfw_L0 5-decade grid).
        original = self.payload.r
        in_range = original[(original > lo) & (original < hi)]
        uniform = np.linspace(lo, hi, num)
        merged = np.unique(np.concatenate([in_range, uniform]))
        return merged

    def _infer_horizons(self) -> list[float]:
        """Scan PCHIP-interpolated ``A`` for zero crossings inside the grid."""
        samples = self._interior_sample_grid()
        if len(samples) < 2:
            return []
        f = self._A_pchip
        roots: list[float] = []
        for bracket in self._scan_brackets(samples, f):
            root = self._refine_root(f, bracket)
            if root is not None and self._r_grid_min < root < self._r_grid_max:
                roots.append(root)
        return sorted(roots)

    def _infer_photon_spheres(self) -> list[float]:
        """Scan ``r * A'(r) - 2 * A(r)`` for zero crossings inside the grid."""
        samples = self._interior_sample_grid()
        if len(samples) < 2:
            return []

        def f(r: float) -> float:
            return float(r * self._dA_pchip(r) - 2.0 * self._A_pchip(r))

        roots: list[float] = []
        for bracket in self._scan_brackets(samples, f):
            root = self._refine_root(f, bracket)
            if root is not None and self._r_grid_min < root < self._r_grid_max:
                roots.append(root)
        return sorted(roots)
