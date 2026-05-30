"""``.npz`` loader and validation gate for numerical static metrics.

This module is the thin orchestration layer that maps raw ``.npz`` files into
the :class:`CanonicalPayload` defined in
:mod:`spherical_raytracing.numerical_metrics` (plan steps 1-2).

Two input formats are accepted (plan decision 7):

- Standard ``r/A/B``: ``A=A``, ``C=1/B``, ``metadata.raw_B is None``.
- Li-DM-BH ``r/f/g/B/params``: ``A=f``, ``C=g``, ``metadata.raw_B`` is the
  original ``g_rr`` array (used for the ``B*C≈1`` soft check).

``ValidationGate`` then enforces the hard / soft thresholds enumerated in
plan decision 8.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .numerical_metrics import (
    CanonicalPayload,
    LiDMBHMetadata,
    ValidationOptions,
)


# Expected convention values for the Li-DM-BH format. Mismatches are hard
# failures per plan decision 8 (``Li conventions conflict``).
_EXPECTED_LI_CONVENTIONS = {
    "g_convention": "g_rr_inverse",
    "B_convention": "g_rr",
}


def load_metric_npz(
    path: str | os.PathLike[str],
    *,
    payload_format: str = "auto",
    validation: ValidationOptions | None = None,
) -> CanonicalPayload:
    """Load a numerical static metric ``.npz`` file into a :class:`CanonicalPayload`.

    The loader is a small, focused step in the pipeline:
    ``.npz`` → :class:`CanonicalPayload` → :class:`ValidationGate` →
    :class:`StaticDomainSelector` → :class:`TabulatedMetric`. The validation
    gate runs before the payload is returned so callers can rely on the
    invariants in plan decision 8 (strict-monotone ``r``, no NaN/inf, Li
    convention agreement, metadata-horizon residuals within hard tolerance).

    Parameters
    ----------
    path:
        ``.npz`` file. Either ``r/A/B`` (standard) or ``r/f/g/B/params``
        (Li-DM-BH) layout is accepted.
    payload_format:
        ``"auto"`` infers from the keys; ``"standard-rab"`` and ``"li-dm-bh"``
        force a specific layout.
    validation:
        Override the default :class:`ValidationOptions` thresholds.

    Returns
    -------
    CanonicalPayload
        Validated payload with ``A=A``, ``C=1/B``, ``metadata`` populated.
        Standard ``r/A/B`` files end up with ``metadata.raw_B is None``.

    See Also
    --------
    spherical_raytracing.numerical_metrics.TabulatedMetric :
        Wrap the returned payload to obtain a metric usable with
        :class:`QuadTransferSolver` and :class:`FiniteStaticObserver`.
    spherical_raytracing.numerical_metrics.StaticDomainSelector :
        Pick the active ``[r_lo, r_hi]`` patch from the validated payload.
    """
    path = Path(path)
    raw = _load_raw_arrays(path)
    inferred_format = _resolve_payload_format(payload_format, raw)
    payload = _build_payload(inferred_format, raw)

    options = validation if validation is not None else ValidationOptions()
    return ValidationGate(options=options).run(payload)


# =====================================================================
# Raw .npz handling
# =====================================================================
def _load_raw_arrays(path: Path) -> dict[str, Any]:
    """Read ``.npz`` keys lazily into a plain dict (closing the archive)."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = list(archive.files)
            data = {key: archive[key] for key in keys}
    except ValueError as exc:
        if "Object arrays cannot be loaded" in str(exc):
            raise ValueError(
                "unsupported object array in metric archive; pickle loading is disabled"
            ) from exc
        raise
    return data


def _resolve_payload_format(payload_format: str, raw: dict[str, Any]) -> str:
    if payload_format == "auto":
        if {"r", "f", "g", "B", "params"}.issubset(raw):
            return "li-dm-bh"
        if {"r", "A", "B"}.issubset(raw):
            return "standard-rab"
        raise ValueError(
            f"cannot infer payload_format from keys={sorted(raw)}; "
            "expected 'r/A/B' or 'r/f/g/B/params'"
        )
    if payload_format == "li-dm-bh":
        missing = {"r", "f", "g", "B", "params"} - set(raw)
        if missing:
            raise ValueError(f"missing Li-DM-BH keys: {sorted(missing)}")
        return payload_format
    if payload_format == "standard-rab":
        missing = {"r", "A", "B"} - set(raw)
        if missing:
            raise ValueError(f"missing standard r/A/B keys: {sorted(missing)}")
        return payload_format
    raise ValueError(
        f"unknown payload_format={payload_format!r}; expected 'auto', 'li-dm-bh', 'standard-rab'"
    )


def _build_payload(payload_format: str, raw: dict[str, Any]) -> CanonicalPayload:
    if payload_format == "li-dm-bh":
        params_raw = raw["params"]
        params = json.loads(_decode_params(params_raw))
        raw_B = np.asarray(raw["B"], dtype=float)
        metadata = LiDMBHMetadata.from_params(
            params=params,
            source_format="li-dm-bh",
            original_keys=("r", "f", "g", "B", "params"),
            raw_B=raw_B,
        )
        r = np.asarray(raw["r"], dtype=float)
        A = np.asarray(raw["f"], dtype=float)
        C = np.asarray(raw["g"], dtype=float)
        return CanonicalPayload(r=r, A=A, C=C, metadata=metadata)

    if payload_format == "standard-rab":
        metadata = LiDMBHMetadata.from_params(
            params=None,
            source_format="standard-rab",
            original_keys=("r", "A", "B"),
            raw_B=None,
        )
        r = np.asarray(raw["r"], dtype=float)
        A = np.asarray(raw["A"], dtype=float)
        B = np.asarray(raw["B"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            C = np.where(B != 0.0, 1.0 / B, np.nan)
        return CanonicalPayload(r=r, A=A, C=C, metadata=metadata)

    raise ValueError(f"unsupported payload_format={payload_format!r}")


def _decode_params(params_raw: Any) -> str:
    if isinstance(params_raw, np.ndarray):
        if params_raw.shape != ():
            raise ValueError(
                f"params must be a scalar JSON string/bytes field, got shape={params_raw.shape}"
            )
        params_raw = params_raw.item()
    if isinstance(params_raw, bytes):
        return params_raw.decode("utf-8")
    if isinstance(params_raw, str):
        return params_raw
    raise ValueError(
        f"params must be a scalar JSON string/bytes field, got {type(params_raw).__name__}"
    )


# =====================================================================
# Validation gate (plan decision 8)
# =====================================================================
class ValidationGate:
    """Run hard / soft checks against a :class:`CanonicalPayload`."""

    def __init__(self, *, options: ValidationOptions | None = None) -> None:
        self.options = options if options is not None else ValidationOptions()

    def run(self, payload: CanonicalPayload) -> CanonicalPayload:
        """Validate ``payload`` in place; return it for fluent chaining."""
        self._hard_checks(payload)
        self._soft_checks(payload)
        return payload

    # ----- hard checks (plan decision 8 hard list) ---------------------
    def _hard_checks(self, payload: CanonicalPayload) -> None:
        r = payload.r
        A = payload.A
        C = payload.C

        if r.ndim != 1 or A.shape != r.shape or C.shape != r.shape:
            raise ValueError(
                f"payload arrays must be 1-D and aligned (r={r.shape}, A={A.shape}, C={C.shape})"
            )
        if np.any(~np.isfinite(r)) or np.any(~np.isfinite(A)) or np.any(~np.isfinite(C)):
            raise ValueError("payload contains NaN/inf entries (r/A/C)")
        if np.any(np.diff(r) <= 0):
            raise ValueError("r must be strictly increasing")

        # Convention conflict (Li-DM-BH only).
        if payload.metadata.source_format == "li-dm-bh":
            for key, expected in _EXPECTED_LI_CONVENTIONS.items():
                actual = payload.metadata.conventions.get(key)
                if actual is None:
                    # Missing convention => soft warning (handled below).
                    continue
                if actual != expected:
                    raise ValueError(
                        f"Li-DM-BH convention conflict: {key}={actual!r}, expected {expected!r}"
                    )

    # ----- soft checks (plan decision 8 soft list) ---------------------
    def _soft_checks(self, payload: CanonicalPayload) -> None:
        opt = self.options
        meta = payload.metadata

        if meta.source_format == "li-dm-bh":
            if meta.raw_B is None:
                warnings.warn(
                    "Li-DM-BH payload missing raw_B; B*C consistency check skipped",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                bc = meta.raw_B * payload.C
                residual = float(np.max(np.abs(bc - 1.0)))
                # Plan decision 8 lists the soft window as ``[1e-12, 1e-8]``;
                # we treat ``residual < bc_warn_lower`` as machine-precision
                # agreement (silent OK) and only warn above ``bc_warn_upper``.
                # This is the pragmatic reading -- real Li-DM-BH files sit at
                # ~1e-16 and would otherwise flood callers with spurious
                # "too good" warnings.
                if residual > opt.bc_warn_upper:
                    warnings.warn(
                        f"|B*C - 1| = {residual:.3e} above soft window upper "
                        f"bound {opt.bc_warn_upper:.1e}",
                        UserWarning,
                        stacklevel=2,
                    )

            missing_conv = [key for key in _EXPECTED_LI_CONVENTIONS if key not in meta.conventions]
            if missing_conv:
                warnings.warn(
                    f"Li-DM-BH metadata missing conventions: {sorted(missing_conv)}",
                    UserWarning,
                    stacklevel=2,
                )

        if not meta.static_domains:
            warnings.warn(
                "metadata.static_domains is empty; static patch will be inferred from A>0 & C>0",
                UserWarning,
                stacklevel=2,
            )

        # Hard residual check: metadata horizons that map to a near-by tabulated
        # ``A`` sign change must agree to within ``horizon_residual_warn_upper``
        # (plan decision 8: hard fail if residual > 1e-4). Horizons that sit
        # below ``r_grid_min`` or above ``r_grid_max`` (i.e. the table starts
        # just inside the static patch with no sign change to anchor against)
        # are handled with a softer rule: the metadata horizon must lie within
        # one grid spacing of the corresponding edge, otherwise it is a hard
        # disagreement.
        if meta.horizons:
            grid_residuals = self._horizon_grid_residuals(payload)
            hard_failed = [
                residual
                for residual in grid_residuals
                if math.isfinite(residual) and residual > opt.horizon_residual_warn_upper
            ]
            if hard_failed:
                raise ValueError(
                    "metadata horizons disagree with tabulated A=0 zeros beyond "
                    f"{opt.horizon_residual_warn_upper:g}: residuals={hard_failed}"
                )
            soft_window = [
                residual
                for residual in grid_residuals
                if math.isfinite(residual)
                and opt.horizon_residual_warn_lower <= residual <= opt.horizon_residual_warn_upper
            ]
            if soft_window:
                warnings.warn(
                    f"metadata horizons residuals in soft window {soft_window}",
                    UserWarning,
                    stacklevel=2,
                )
            unmatched = [
                horizon
                for horizon, residual in zip(meta.horizons, grid_residuals)
                if not math.isfinite(residual)
            ]
            for horizon in unmatched:
                self._check_unmatched_horizon_at_edge(payload, horizon)

        # Photon-sphere residual: metadata ``photon_spheres`` should agree with
        # zero crossings of ``r*A' - 2A`` on the tabulated grid to within
        # ``photon_sphere_residual_warn`` (plan decision 8 soft list).
        if meta.photon_spheres:
            ps_residuals = self._photon_sphere_grid_residuals(payload)
            offending = [
                (rps, residual)
                for rps, residual in zip(meta.photon_spheres, ps_residuals)
                if math.isfinite(residual) and residual > opt.photon_sphere_residual_warn
            ]
            if offending:
                warnings.warn(
                    "metadata photon_spheres disagree with tabulated rA'-2A=0 zeros: "
                    f"{offending}",
                    UserWarning,
                    stacklevel=2,
                )

        # Plan decision 8 soft list (last item): warn that a finite-grid table
        # is not suitable for ``InfinityObserver``. This is a hint -- the hard
        # block lives in the solver guard (plan decision 11); we surface it at
        # load time so users wiring up a pipeline catch the mismatch early.
        r_grid_max = float(np.max(payload.r))
        unbounded_domain = any(
            (math.isinf(domain[1]) or domain[1] >= r_grid_max)
            for domain in meta.static_domains
        )
        if unbounded_domain and math.isfinite(r_grid_max):
            warnings.warn(
                f"table r_max={r_grid_max:g} is finite; pair TabulatedMetric "
                "with FiniteStaticObserver (InfinityObserver is hard-blocked)",
                UserWarning,
                stacklevel=2,
            )

    def _check_unmatched_horizon_at_edge(
        self,
        payload: CanonicalPayload,
        horizon: float,
    ) -> None:
        """A metadata horizon with no in-grid sign change must sit at a grid edge.

        Some Li-DM-BH files (e.g. ``hernquist_L0.npz``) sample the static patch
        starting just above the horizon, so the table has no zero crossing
        despite the metadata horizon being authoritative. We accept this only
        when the metadata horizon is within one grid spacing of the closest
        grid endpoint; otherwise we hard-fail with a clear message.
        """
        r = payload.r
        edge_lo = float(r[0])
        edge_hi = float(r[-1])
        spacing_lo = float(r[1] - r[0]) if len(r) >= 2 else math.inf
        spacing_hi = float(r[-1] - r[-2]) if len(r) >= 2 else math.inf
        gap_lo = abs(horizon - edge_lo)
        gap_hi = abs(horizon - edge_hi)
        if gap_lo <= spacing_lo or gap_hi <= spacing_hi:
            return
        raise ValueError(
            "metadata horizon "
            f"{horizon!r} has no nearby tabulated A=0 sign change and is "
            f"more than one grid spacing from either edge "
            f"(grid edges=({edge_lo}, {edge_hi}), spacings=({spacing_lo}, {spacing_hi}))"
        )

    @staticmethod
    def _horizon_grid_residuals(payload: CanonicalPayload) -> list[float]:
        """Approximate residual = distance from each metadata horizon to the nearest sign change of A."""
        r = payload.r
        A = payload.A
        sign_changes: list[float] = []
        for i in range(len(r) - 1):
            if A[i] == 0.0:
                sign_changes.append(float(r[i]))
            elif A[i] * A[i + 1] < 0.0:
                # Linear interpolation root estimate.
                ri = r[i] + (r[i + 1] - r[i]) * A[i] / (A[i] - A[i + 1])
                sign_changes.append(float(ri))
        if A[-1] == 0.0:
            sign_changes.append(float(r[-1]))

        residuals: list[float] = []
        for h in payload.metadata.horizons:
            if not sign_changes:
                residuals.append(math.inf)
                continue
            residual = min(abs(h - sc) for sc in sign_changes)
            residuals.append(residual)
        return residuals

    @staticmethod
    def _photon_sphere_grid_residuals(payload: CanonicalPayload) -> list[float]:
        """Distance from each metadata ``r_ph`` to the nearest zero of ``rA'-2A`` on the grid.

        ``rA' - 2A = 0`` is the photon-sphere defining equation for a static
        spherically symmetric metric. We approximate ``A'`` with a centred
        second-order finite difference, scan for sign changes of ``r*A' - 2A``,
        and report ``min(|r_ph - root|)`` per metadata entry. ``inf`` when no
        sign change exists on the grid (the metadata photon sphere may sit
        outside the sampled support).
        """
        r = np.asarray(payload.r, dtype=float)
        A = np.asarray(payload.A, dtype=float)
        if len(r) < 3:
            return [math.inf] * len(payload.metadata.photon_spheres)
        dA_dr = np.gradient(A, r, edge_order=2)
        f = r * dA_dr - 2.0 * A
        roots: list[float] = []
        for i in range(len(r) - 1):
            if f[i] == 0.0:
                roots.append(float(r[i]))
            elif f[i] * f[i + 1] < 0.0:
                # Linear interpolation root estimate.
                ri = r[i] + (r[i + 1] - r[i]) * f[i] / (f[i] - f[i + 1])
                roots.append(float(ri))
        if f[-1] == 0.0:
            roots.append(float(r[-1]))

        residuals: list[float] = []
        for r_ph in payload.metadata.photon_spheres:
            if not roots:
                residuals.append(math.inf)
                continue
            residuals.append(min(abs(r_ph - root) for root in roots))
        return residuals
