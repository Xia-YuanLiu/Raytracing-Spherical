"""TDD tracer bullets for the numerical metric ingest pipeline.

Each test exercises one independent slice of the plan in
``docs/plans/数值度规接入计划.md``:

- LiDMBHMetadata schema (TB1)
- CanonicalPayload schema (TB2)
- load_metric_npz on both formats (TB3)
- ValidationGate hard failures (TB4)
- ValidationGate soft warnings (TB5)
- StaticDomainSelector (TB6)
- TabulatedMetric A/B/G basic behaviour (TB7)
- PCHIP endpoint-zero anchoring (TB8)
- Schwarzschild round-trip (TB9)
- TabulatedMetric + InfinityObserver / ThroughTracePolicy hard raise (TB10)
- Seven real ``.npz`` end-to-end smoke (TB11)
- ``burkert_Lneq`` thin-disk profile smoke (TB12)
"""

from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pytest


LI_DATA_DIR = Path("/home/yuanpang/Work/Li-DM-BH/outputs/sanity/data")


def _make_li_params_dict(**overrides):
    base = {
        "model": "burkert",
        "MB": 1.0,
        "Lambda": 0.01,
        "a": 5.0,
        "M": 1.0,
        "mode": "Lneq",
        "static_domains": [[2.0277939461512866, 15.865581178272851]],
        "horizons": [2.0277939461512866, 15.865581178272851],
        "photon_spheres": [3.0168216613005847],
        "g_convention": "g_rr_inverse",
        "B_convention": "g_rr",
    }
    base.update(overrides)
    return base


# =====================================================================
# Tracer bullet 1: LiDMBHMetadata dataclass
# =====================================================================
def test_li_dm_bh_metadata_parses_params_dict_with_typed_fields():
    """LiDMBHMetadata accepts the params dict from Li-DM-BH npz and exposes typed fields."""
    from spherical_raytracing.numerical_metrics import LiDMBHMetadata

    params = _make_li_params_dict()
    raw_b = np.array([1.0, 2.0, 3.0])

    metadata = LiDMBHMetadata.from_params(
        params=params,
        source_format="li-dm-bh",
        original_keys=("r", "f", "g", "B", "params"),
        raw_B=raw_b,
    )

    assert metadata.source_format == "li-dm-bh"
    assert metadata.original_keys == ("r", "f", "g", "B", "params")
    assert metadata.static_domains == ((2.0277939461512866, 15.865581178272851),)
    assert metadata.horizons == (2.0277939461512866, 15.865581178272851)
    assert metadata.photon_spheres == (3.0168216613005847,)
    assert metadata.conventions == {"g_convention": "g_rr_inverse", "B_convention": "g_rr"}
    assert metadata.model_params["model"] == "burkert"
    assert metadata.model_params["MB"] == 1.0
    assert np.array_equal(metadata.raw_B, raw_b)


# =====================================================================
# Tracer bullet 2: CanonicalPayload dataclass
# =====================================================================
def test_canonical_payload_holds_canonical_arrays_with_optional_raw_b():
    """CanonicalPayload bundles r/A/C plus metadata; raw_B is on metadata only."""
    from spherical_raytracing.numerical_metrics import CanonicalPayload, LiDMBHMetadata

    r = np.linspace(2.1, 10.0, 5)
    A = 1.0 - 2.0 / r
    C = A.copy()
    metadata = LiDMBHMetadata.from_params(
        params=None,
        source_format="standard-rab",
        original_keys=("r", "A", "B"),
        raw_B=None,
    )
    payload = CanonicalPayload(r=r, A=A, C=C, metadata=metadata)

    assert np.array_equal(payload.r, r)
    assert np.array_equal(payload.A, A)
    assert np.array_equal(payload.C, C)
    assert payload.metadata.raw_B is None
    assert payload.metadata.source_format == "standard-rab"


# =====================================================================
# Tracer bullet 3: load_metric_npz supports both formats
# =====================================================================
def _skip_if_li_data_missing():
    if not LI_DATA_DIR.exists():
        pytest.skip(f"Li-DM-BH data directory not available at {LI_DATA_DIR}")


def test_load_metric_npz_handles_li_dm_bh_format(tmp_path):
    """A Li-DM-BH ``r/f/g/B/params`` file yields a CanonicalPayload with raw_B preserved."""
    _skip_if_li_data_missing()
    from spherical_raytracing import CanonicalPayload, load_metric_npz

    path = LI_DATA_DIR / "burkert_Lneq.npz"
    payload = load_metric_npz(path)

    assert isinstance(payload, CanonicalPayload)
    assert payload.metadata.source_format == "li-dm-bh"
    assert payload.metadata.raw_B is not None
    assert payload.metadata.raw_B.shape == payload.r.shape
    assert payload.metadata.conventions["g_convention"] == "g_rr_inverse"
    assert payload.metadata.conventions["B_convention"] == "g_rr"
    assert payload.metadata.static_domains, "Li-DM-BH file must carry static_domains"
    np.testing.assert_allclose(payload.C, 1.0 / payload.metadata.raw_B, rtol=0, atol=1e-12)


def test_load_metric_npz_handles_standard_rab_format(tmp_path):
    """A bare ``r/A/B`` payload yields raw_B is None and C == 1/B."""
    from spherical_raytracing import load_metric_npz

    r = np.linspace(2.1, 50.0, 200)
    A = 1.0 - 2.0 / r
    B = 1.0 / A
    path = tmp_path / "schwarzschild_rab.npz"
    np.savez(path, r=r, A=A, B=B)

    payload = load_metric_npz(path)

    assert payload.metadata.source_format == "standard-rab"
    assert payload.metadata.raw_B is None
    np.testing.assert_allclose(payload.C, 1.0 / B, rtol=0, atol=1e-15)
    np.testing.assert_allclose(payload.A, A, rtol=0, atol=1e-15)


def test_load_metric_npz_rejects_object_arrays_without_pickle(tmp_path):
    """Object arrays in user-supplied archives must fail validation without
    enabling pickle loading.
    """
    from spherical_raytracing import load_metric_npz

    r = np.linspace(2.1, 50.0, 10)
    A = 1.0 - 2.0 / r
    path = tmp_path / "object_payload.npz"
    np.savez(path, r=r, A=A, B=np.array([{"malicious": "payload"}], dtype=object))

    with pytest.raises(ValueError, match="object array|pickle loading is disabled"):
        load_metric_npz(path, payload_format="standard-rab")


def test_load_metric_npz_decodes_scalar_bytes_params(tmp_path):
    """Li-DM-BH ``params`` may be saved as a scalar byte string; ``np.load``
    returns a 0-D ndarray that must be unwrapped before JSON decoding.
    """
    from spherical_raytracing import load_metric_npz

    r = np.linspace(2.1, 10.0, 50)
    f = 1.0 - 2.0 / r
    g = f.copy()
    B = 1.0 / g
    params = _make_li_params_dict(
        horizons=[],
        photon_spheres=[],
        static_domains=[[2.1, 10.0]],
    )
    path = tmp_path / "li_scalar_bytes_params.npz"
    np.savez(path, r=r, f=f, g=g, B=B, params=json.dumps(params).encode("utf-8"))

    payload = load_metric_npz(path, payload_format="li-dm-bh")

    assert payload.metadata.source_format == "li-dm-bh"
    assert payload.metadata.model_params["model"] == "burkert"
    assert payload.metadata.static_domains == ((2.1, 10.0),)


# =====================================================================
# Tracer bullet 4: ValidationGate hard failures
# =====================================================================
def _make_li_payload(r, A, C, params_overrides=None, raw_B=None):
    from spherical_raytracing.numerical_metrics import CanonicalPayload, LiDMBHMetadata

    params = _make_li_params_dict(**(params_overrides or {}))
    C_arr = np.asarray(C)
    if raw_B is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_B = np.where(C_arr != 0.0, 1.0 / C_arr, np.nan)
    metadata = LiDMBHMetadata.from_params(
        params=params,
        source_format="li-dm-bh",
        original_keys=("r", "f", "g", "B", "params"),
        raw_B=raw_B,
    )
    return CanonicalPayload(r=np.asarray(r), A=np.asarray(A), C=C_arr, metadata=metadata)


def test_validation_gate_rejects_non_monotone_r():
    from spherical_raytracing import ValidationGate

    r = np.array([2.5, 2.4, 3.0, 4.0])
    A = np.array([0.1, 0.2, 0.3, 0.4])
    C = A.copy()
    payload = _make_li_payload(r, A, C)

    with pytest.raises(ValueError, match="strictly increasing"):
        ValidationGate().run(payload)


def test_validation_gate_rejects_nan_in_arrays():
    from spherical_raytracing import ValidationGate

    r = np.linspace(2.1, 10.0, 5)
    A = np.array([0.1, np.nan, 0.3, 0.4, 0.5])
    C = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    payload = _make_li_payload(r, A, C)

    with pytest.raises(ValueError, match="NaN/inf"):
        ValidationGate().run(payload)


def test_validation_gate_rejects_li_dm_bh_convention_mismatch():
    from spherical_raytracing import ValidationGate

    r = np.linspace(2.1, 10.0, 5)
    A = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    C = A.copy()
    payload = _make_li_payload(
        r, A, C,
        params_overrides={"g_convention": "g_rr", "B_convention": "g_rr_inverse"},
    )

    with pytest.raises(ValueError, match="convention conflict"):
        ValidationGate().run(payload)


def test_load_metric_npz_rejects_missing_required_keys(tmp_path):
    """Forcing a layout against an incomplete archive raises ``ValueError``."""
    from spherical_raytracing import load_metric_npz

    path = tmp_path / "bad.npz"
    np.savez(path, r=np.linspace(2.0, 10.0, 5))  # missing A and B

    with pytest.raises(ValueError, match="cannot infer payload_format|missing"):
        load_metric_npz(path)


def test_validation_gate_rejects_horizon_residual_beyond_hard_threshold():
    """Metadata horizons disagreeing with the tabulated A=0 zero beyond 1e-4 is a hard failure."""
    from spherical_raytracing import ValidationGate

    r = np.linspace(2.0, 10.0, 9)
    A = r - 2.0  # zero exactly at r=2.0
    C = A.copy()
    payload = _make_li_payload(
        r, A, C,
        params_overrides={"horizons": [3.5], "static_domains": [[3.5, 10.0]]},
    )

    with pytest.raises(ValueError, match="metadata horizons"):
        ValidationGate().run(payload)


# =====================================================================
# Tracer bullet 5: ValidationGate soft warnings
# =====================================================================
def test_validation_gate_warns_on_bc_consistency_above_threshold():
    """|B*C - 1| > 1e-8 (with raw_B explicitly provided) raises a soft warning."""
    from spherical_raytracing import ValidationGate

    r = np.linspace(2.1, 10.0, 5)
    A = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    C = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    raw_B = (1.0 / C) * (1.0 + 1e-6)
    payload = _make_li_payload(
        r, A, C,
        params_overrides={"horizons": [], "photon_spheres": [], "static_domains": [[2.1, 10.0]]},
        raw_B=raw_B,
    )

    with pytest.warns(UserWarning, match=r"\|B\*C - 1\|"):
        ValidationGate().run(payload)


def test_validation_gate_warns_on_horizon_residual_in_soft_window():
    """A horizon residual ~1e-5 vs the tabulated A-zero falls in the soft window [1e-8, 1e-4]."""
    from spherical_raytracing import ValidationGate

    r = np.linspace(2.0, 10.0, 9)
    A = r - 2.0
    C = A.copy()
    # Metadata horizon offset by 5e-5 from the real zero.
    payload = _make_li_payload(
        r, A, C,
        params_overrides={"horizons": [2.0 + 5e-5], "static_domains": [[2.0 + 5e-5, 10.0]]},
    )

    with pytest.warns(UserWarning, match="soft window"):
        ValidationGate().run(payload)


# =====================================================================
# Tracer bullet 6: StaticDomainSelector
# =====================================================================
def test_static_domain_selector_default_picks_last_metadata_domain():
    """Default ``domain_index=-1`` picks the last entry of metadata.static_domains."""
    from spherical_raytracing import StaticDomainSelector

    r = np.linspace(0.5, 20.0, 200)
    A = np.ones_like(r) * 0.5
    C = A.copy()
    payload = _make_li_payload(
        r, A, C,
        params_overrides={
            "horizons": [],
            "photon_spheres": [],
            "static_domains": [[0.5, 1.5], [2.0, 18.0]],
        },
    )

    r_lo, r_hi = StaticDomainSelector().choose(payload)
    assert r_lo == 2.0
    assert r_hi == 18.0


def test_static_domain_selector_radial_bounds_override_metadata():
    """Explicit ``radial_bounds`` win over ``metadata.static_domains``."""
    from spherical_raytracing import StaticDomainSelector

    r = np.linspace(0.5, 20.0, 200)
    A = np.ones_like(r) * 0.5
    C = A.copy()
    payload = _make_li_payload(
        r, A, C,
        params_overrides={
            "horizons": [],
            "photon_spheres": [],
            "static_domains": [[2.0, 18.0]],
        },
    )

    r_lo, r_hi = StaticDomainSelector(radial_bounds=(4.0, 12.0)).choose(payload)
    assert r_lo == 4.0
    assert r_hi == 12.0


def test_static_domain_selector_maps_null_upper_to_infinity():
    """A metadata upper bound of ``None`` (JSON null) maps to ``math.inf``."""
    from spherical_raytracing.numerical_metrics import LiDMBHMetadata

    metadata = LiDMBHMetadata.from_params(
        params=_make_li_params_dict(
            horizons=[2.0],
            photon_spheres=[3.0],
            static_domains=[[2.0, None]],
        ),
        source_format="li-dm-bh",
        original_keys=("r", "f", "g", "B", "params"),
        raw_B=None,
    )
    assert metadata.static_domains == ((2.0, math.inf),)


# =====================================================================
# Tracer bullet 7: TabulatedMetric A/B/G basic behaviour + domain raises
# =====================================================================
def _build_schwarzschild_payload(r_lo=2.0, r_hi=200.0, num=400):
    from spherical_raytracing.numerical_metrics import CanonicalPayload, LiDMBHMetadata

    r = np.geomspace(r_lo + 1e-3, r_hi, num)
    A = 1.0 - 2.0 / r
    C = A.copy()
    metadata = LiDMBHMetadata.from_params(
        params={
            "model": "schwarzschild",
            "static_domains": [[r_lo, None]],
            "horizons": [r_lo],
            "photon_spheres": [3.0],
            "g_convention": "g_rr_inverse",
            "B_convention": "g_rr",
        },
        source_format="li-dm-bh",
        original_keys=("r", "f", "g", "B", "params"),
        raw_B=1.0 / C,
    )
    return CanonicalPayload(r=r, A=A, C=C, metadata=metadata)


def test_tabulated_metric_returns_finite_A_B_G_inside_static_domain():
    from spherical_raytracing import TabulatedMetric

    payload = _build_schwarzschild_payload()
    metric = TabulatedMetric(payload=payload, static_domain=(2.0, math.inf))

    r = 10.0
    expected_A = 1.0 - 2.0 / r
    assert math.isclose(metric.A(r), expected_A, rel_tol=1e-3)
    assert math.isclose(metric.B(r), 1.0 / expected_A, rel_tol=1e-3)

    u = 1.0 / r
    b = 7.0
    g = metric.G(u, b)
    assert math.isfinite(g)


def test_tabulated_metric_raises_outside_static_domain():
    from spherical_raytracing import TabulatedMetric

    payload = _build_schwarzschild_payload(r_lo=2.0, r_hi=50.0)
    metric = TabulatedMetric(payload=payload, static_domain=(2.0, 40.0))

    with pytest.raises(ValueError, match="outside static domain"):
        metric.A(1.5)
    with pytest.raises(ValueError, match="outside static domain"):
        metric.A(45.0)


def test_tabulated_metric_raises_when_c_below_tolerance():
    """``B(r)`` and ``dB_dr(r)`` raise when ``|C| < c_tol``."""
    from spherical_raytracing.numerical_metrics import CanonicalPayload, LiDMBHMetadata, TabulatedMetric

    # Construct a payload where C decreases towards a small value at a tiny r.
    r = np.linspace(2.001, 10.0, 200)
    A = 1.0 - 2.0 / r
    C = A.copy()
    metadata = LiDMBHMetadata.from_params(
        params={
            "static_domains": [[2.0, None]],
            "horizons": [2.0],
            "photon_spheres": [3.0],
            "g_convention": "g_rr_inverse",
            "B_convention": "g_rr",
        },
        source_format="li-dm-bh",
        original_keys=("r", "f", "g", "B", "params"),
        raw_B=1.0 / C,
    )
    payload = CanonicalPayload(r=r, A=A, C=C, metadata=metadata)
    metric = TabulatedMetric(payload=payload, static_domain=(2.0, math.inf))

    r_near_horizon = 2.0 + 1e-13
    with pytest.raises(ValueError, match="singular"):
        metric.B(r_near_horizon)


def test_tabulated_metric_G_raises_when_u_below_grid_support():
    """``G(u, b)`` with ``u→0`` (i.e. ``r→∞``) must raise outside the grid support."""
    from spherical_raytracing import TabulatedMetric

    payload = _build_schwarzschild_payload(r_lo=2.0, r_hi=50.0)
    metric = TabulatedMetric(payload=payload, static_domain=(2.0, math.inf))

    with pytest.raises(ValueError):
        metric.G(u=0.0, b=5.0)  # r = infinity
    with pytest.raises(ValueError):
        metric.G(u=1.0 / 1000.0, b=5.0)  # r = 1000 > r_grid_max = 50


# =====================================================================
# Tracer bullet 8: PCHIP endpoint-zero anchor (A and C vanish at the boundary)
# =====================================================================
def test_tabulated_metric_anchors_A_and_C_to_zero_at_finite_boundaries():
    """``A(r_h+ε) → 0`` and ``A(r_c-ε) → 0`` thanks to the (r_boundary, 0) anchors."""
    from spherical_raytracing.numerical_metrics import CanonicalPayload, LiDMBHMetadata, TabulatedMetric

    # Synthetic bounded domain mimicking RNdS: A=(r-r_h)*(r_c-r)/(r_h*r_c)
    r_h, r_c = 2.0, 16.0
    r = np.linspace(r_h + 1e-3, r_c - 1e-3, 400)
    A = (r - r_h) * (r_c - r) / (r_h * r_c)
    C = A.copy()
    metadata = LiDMBHMetadata.from_params(
        params={
            "static_domains": [[r_h, r_c]],
            "horizons": [r_h, r_c],
            "photon_spheres": [3.5],
            "g_convention": "g_rr_inverse",
            "B_convention": "g_rr",
        },
        source_format="li-dm-bh",
        original_keys=("r", "f", "g", "B", "params"),
        raw_B=1.0 / C,
    )
    payload = CanonicalPayload(r=r, A=A, C=C, metadata=metadata)
    metric = TabulatedMetric(payload=payload, static_domain=(r_h, r_c))

    near_h = r_h + 1e-6
    near_c = r_c - 1e-6
    assert abs(metric.A(near_h)) < 1e-5
    assert abs(metric.A(near_c)) < 1e-5
    # The interpolant is anchored at the boundaries; far inside the domain the
    # original physical value is recovered.
    mid = 0.5 * (r_h + r_c)
    assert math.isclose(metric.A(mid), (mid - r_h) * (r_c - mid) / (r_h * r_c), rel_tol=1e-3)


# =====================================================================
# Tracer bullet 9: Schwarzschild round-trip
# =====================================================================
def test_tabulated_metric_schwarzschild_round_trip_matches_analytic_phi_within_1e8():
    """A TabulatedMetric fed with the analytic Schwarzschild table reproduces the
    analytic ray trace to within ``1e-8`` relative error across a representative
    sweep of impact parameters (plan decision 4 acceptance)."""
    from spherical_raytracing import (
        FiniteStaticObserver,
        QuadTransferSolver,
        SchwarzschildMetric,
        TabulatedMetric,
    )

    payload = _build_schwarzschild_payload(r_lo=2.0, r_hi=500.0, num=4000)
    sch = SchwarzschildMetric(mass=1.0)
    tab = TabulatedMetric(payload=payload, static_domain=(2.0, math.inf))

    obs_a = FiniteStaticObserver(r_obs=100.0, metric=sch)
    obs_t = FiniteStaticObserver(r_obs=100.0, metric=tab)
    solver_a = QuadTransferSolver(metric=sch, observer=obs_a)
    solver_t = QuadTransferSolver(metric=tab, observer=obs_t)

    b_crit = sch.critical_curves()[0].b_crit  # ≈ 5.196
    for b in [4.0, 5.5, 8.0, 20.0, 80.0]:
        result_a = solver_a.trace_b(b)
        result_t = solver_t.trace_b(b)
        last_a = result_a.segments[-1]
        last_t = result_t.segments[-1]
        relative = abs(last_a.phi_end - last_t.phi_end) / max(abs(last_a.phi_end), 1e-12)
        assert relative < 1e-8, (
            f"phi_end relative diff {relative:.2e} > 1e-8 at b={b} "
            f"(analytic={last_a.phi_end}, tabulated={last_t.phi_end})"
        )
    # Sanity: b_crit produces a sensible result on both backends.
    assert b_crit > 5.0


# =====================================================================
# Tracer bullet 10: TabulatedMetric + InfinityObserver / ThroughTracePolicy raise
# =====================================================================
def test_quad_solver_tabulated_metric_with_infinity_observer_raises():
    """``TabulatedMetric + InfinityObserver`` is unsupported (plan decision 11)."""
    from spherical_raytracing import (
        InfinityObserver,
        QuadTransferSolver,
        TabulatedMetric,
    )

    payload = _build_schwarzschild_payload(r_lo=2.0, r_hi=500.0, num=400)
    metric = TabulatedMetric(payload=payload, static_domain=(2.0, math.inf))
    solver = QuadTransferSolver(metric=metric, observer=InfinityObserver())

    with pytest.raises(NotImplementedError, match="FiniteStaticObserver|InfinityObserver"):
        solver.trace_b(8.0)


def test_quad_solver_tabulated_metric_with_through_policy_raises():
    """``TabulatedMetric + ThroughTracePolicy`` is unsupported (plan decision 10)."""
    from spherical_raytracing import (
        FiniteStaticObserver,
        QuadTransferSolver,
        TabulatedMetric,
        ThroughTracePolicy,
    )
    from spherical_raytracing.diagnostics import EventType

    payload = _build_schwarzschild_payload(r_lo=2.0, r_hi=500.0, num=400)
    metric = TabulatedMetric(payload=payload, static_domain=(2.0, math.inf))
    observer = FiniteStaticObserver(r_obs=100.0, metric=metric)
    policy = ThroughTracePolicy(region_transitions={("external", EventType.HORIZON): "companion"})
    solver = QuadTransferSolver(metric=metric, observer=observer, trace_policy=policy)

    with pytest.raises(NotImplementedError, match="ThroughTracePolicy"):
        solver.trace_b(4.0)


# =====================================================================
# Tracer bullet 11: real .npz end-to-end smoke (7 files)
# =====================================================================
LI_FILES = [
    "burkert_Lneq.npz",
    "hernquist_L0.npz",
    "hernquist_Lneq.npz",
    "jaffe_Lneq.npz",
    "nfw_L0.npz",
    "nfw_Lneq.npz",
    "taylor_silk_Lneq.npz",
]


@pytest.fixture(scope="session")
def li_dm_bh_payloads():
    """Shared, lazily-loaded payloads for all seven Li-DM-BH ``.npz`` files."""
    if not LI_DATA_DIR.exists():
        pytest.skip(f"Li-DM-BH data directory not available at {LI_DATA_DIR}")
    from spherical_raytracing import load_metric_npz

    payloads = {}
    for name in LI_FILES:
        path = LI_DATA_DIR / name
        if not path.exists():
            pytest.skip(f"missing real data file: {path}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            payloads[name] = load_metric_npz(path)
    return payloads


@pytest.mark.parametrize("filename", LI_FILES)
def test_real_li_dm_bh_file_end_to_end_smoke(li_dm_bh_payloads, filename):
    """Each real file loads → TabulatedMetric → FiniteStaticObserver(r_mid) → Quad trace.

    Acceptance per plan step 5: ``phi_end > 0``, ``r_end != r_obs``, no NaN, and
    the inward / outbound endpoints fall on a sensible boundary event
    (``HORIZON`` / ``INNER_BOUNDARY`` / ``OUTER_BOUNDARY`` / ``ESCAPE``).
    """
    from spherical_raytracing import (
        FiniteStaticObserver,
        QuadTransferSolver,
        StaticDomainSelector,
        TabulatedMetric,
    )
    from spherical_raytracing.diagnostics import EventType

    payload = li_dm_bh_payloads[filename]
    r_lo, r_hi = StaticDomainSelector().choose(payload)
    metric = TabulatedMetric(payload=payload, static_domain=(r_lo, r_hi))

    # Choose r_obs at a sensible interior point.
    if math.isfinite(r_hi):
        # Bounded domain: pick midpoint between photon sphere and outer boundary.
        if metric.photon_spheres():
            r_obs = 0.5 * (metric.photon_spheres()[-1] + r_hi)
        else:
            r_obs = 0.5 * (r_lo + r_hi)
    else:
        # Unbounded domain: pick 5×r_ph but stay safely inside the grid.
        r_grid_max = float(np.max(payload.r))
        r_ph = metric.photon_spheres()[-1] if metric.photon_spheres() else 5.0
        r_obs = min(5.0 * r_ph, r_grid_max * 0.9)

    observer = FiniteStaticObserver(r_obs=r_obs, metric=metric)
    solver = QuadTransferSolver(metric=metric, observer=observer)

    crits = metric.critical_curves()
    assert crits, f"{filename}: critical_curves() empty"
    b = crits[0].b_crit * 1.05

    result = solver.trace_b(b)
    assert result.segments, f"{filename}: solver returned no segments"
    last = result.segments[-1]
    assert math.isfinite(last.phi_end), f"{filename}: phi_end is NaN"
    assert last.phi_end > 0.0, f"{filename}: phi_end={last.phi_end} not positive"

    r_end = 1.0 / last.u_end if last.u_end > 0.0 else math.inf
    if math.isfinite(r_hi):
        # Bounded patch (Λ≠0): outbound must reach the outer boundary, not
        # mirror back to r_obs. This guards against the pre-fix mirror bug.
        assert not math.isclose(r_end, r_obs, rel_tol=1e-3, abs_tol=1e-3), (
            f"{filename}: outbound ended at observer radius r_obs={r_obs}, "
            f"the mirror bug regressed"
        )

    accepted_events = {
        EventType.HORIZON,
        EventType.INNER_BOUNDARY,
        EventType.OUTER_BOUNDARY,
        EventType.ESCAPE,
        EventType.TURNING_POINT,
    }
    assert last.endpoint_event in accepted_events, (
        f"{filename}: unexpected endpoint event {last.endpoint_event}"
    )


# =====================================================================
# Tracer bullet 12: burkert_Lneq thin-disk profile smoke
# =====================================================================
def test_burkert_lneq_thin_disk_profile_has_photon_ring_double_intersection(
    li_dm_bh_payloads,
):
    """On ``burkert_Lneq.npz``, a small ``b`` sweep above ``b_crit`` exposes the
    photon-ring double crossing (``intersection_count >= 2``).

    This guards the outbound segment's interaction with the disk window: the
    pre-fix mirror outbound could not reach the second disk crossing because it
    flipped back to ``r_obs`` before crossing the disk again.
    """
    from spherical_raytracing import (
        DiskWindow,
        FiniteStaticObserver,
        QuadTransferSolver,
        StaticDomainSelector,
        TabulatedMetric,
        compute_intersections,
    )

    payload = li_dm_bh_payloads["burkert_Lneq.npz"]
    r_lo, r_hi = StaticDomainSelector().choose(payload)
    metric = TabulatedMetric(payload=payload, static_domain=(r_lo, r_hi))
    r_ph = metric.photon_spheres()[-1]
    r_obs = 0.5 * (r_ph + r_hi)
    observer = FiniteStaticObserver(r_obs=r_obs, metric=metric)
    solver = QuadTransferSolver(metric=metric, observer=observer)

    b_crit = metric.critical_curves()[0].b_crit
    disk = DiskWindow(
        r_min=max(r_ph * 1.05, r_lo + 0.5),
        r_max=min(r_hi - 0.5, r_obs - 0.5),
    )

    # Sweep a small grid of impact parameters just above b_crit.
    b_grid = np.linspace(b_crit * 1.01, b_crit * 1.10, 6)
    max_intersection_count = 0
    for b in b_grid:
        result = solver.trace_b(float(b))
        if not result.segments:
            continue
        records = compute_intersections(result, disk, max_order=4)
        max_intersection_count = max(max_intersection_count, len(records))
    assert max_intersection_count >= 2, (
        f"expected at least one b near b_crit to give intersection_count >= 2 "
        f"(photon ring double crossing); observed max={max_intersection_count}"
    )


# =====================================================================
# Additional coverage tests added in review pass
# =====================================================================
def test_quad_solver_tabulated_metric_rejects_b_above_b_max():
    """``FiniteStaticObserver + b > r_obs / sqrt(A(r_obs))`` is a hard ValueError.

    Plan step 5 / decision 6 hard-fail item; we exercise it on a TabulatedMetric
    backed by a Schwarzschild payload so the b_max threshold is analytically
    predictable.
    """
    from spherical_raytracing import (
        FiniteStaticObserver,
        QuadTransferSolver,
        TabulatedMetric,
    )

    payload = _build_schwarzschild_payload(r_lo=2.0, r_hi=500.0, num=2000)
    metric = TabulatedMetric(payload=payload, static_domain=(2.0, math.inf))
    observer = FiniteStaticObserver(r_obs=100.0, metric=metric)
    solver = QuadTransferSolver(metric=metric, observer=observer)

    b_max = 100.0 / math.sqrt(1.0 - 2.0 / 100.0)
    with pytest.raises(ValueError, match="screen radius"):
        solver.trace_b(b_max * 1.001)


def test_validation_gate_standard_rab_emits_no_bc_warning(tmp_path):
    """Standard ``r/A/B`` files have ``raw_B is None`` so the ``B*C`` soft check
    must be skipped (plan decision 7); only the unrelated "static_domains empty"
    warning is acceptable.
    """
    from spherical_raytracing import load_metric_npz

    r = np.linspace(2.1, 50.0, 200)
    A = 1.0 - 2.0 / r
    B = 1.0 / A
    path = tmp_path / "schwarzschild_rab.npz"
    np.savez(path, r=r, A=A, B=B)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_metric_npz(path)
    bc_warnings = [w for w in caught if "B*C" in str(w.message)]
    assert not bc_warnings, (
        f"standard r/A/B path must not emit B*C warnings, got {bc_warnings}"
    )


def test_tabulated_metric_rejects_negative_a_in_static_domain():
    """A payload with ``A<=0`` inside the selected domain hard-fails at
    construction (plan decision 8 hard list)."""
    from spherical_raytracing.numerical_metrics import (
        CanonicalPayload,
        LiDMBHMetadata,
        TabulatedMetric,
    )

    r = np.linspace(2.1, 10.0, 50)
    A = 1.0 - 2.0 / r
    A[20] = -0.05  # poison one interior sample
    C = np.abs(A) + 0.01  # keep C positive so we test the A check in isolation
    metadata = LiDMBHMetadata.from_params(
        params={
            "static_domains": [[2.0, None]],
            "horizons": [2.0],
            "photon_spheres": [3.0],
            "g_convention": "g_rr_inverse",
            "B_convention": "g_rr",
        },
        source_format="li-dm-bh",
        original_keys=("r", "f", "g", "B", "params"),
        raw_B=1.0 / C,
    )
    payload = CanonicalPayload(r=r, A=A, C=C, metadata=metadata)
    with pytest.raises(ValueError, match="non-positive"):
        TabulatedMetric(payload=payload, static_domain=(2.0, math.inf))


def test_tabulated_metric_rejects_negative_c_in_static_domain():
    """Counterpart of the A check for C (plan decision 8 hard list)."""
    from spherical_raytracing.numerical_metrics import (
        CanonicalPayload,
        LiDMBHMetadata,
        TabulatedMetric,
    )

    r = np.linspace(2.1, 10.0, 50)
    A = 1.0 - 2.0 / r
    C = A.copy()
    C[15] = -0.02
    metadata = LiDMBHMetadata.from_params(
        params={
            "static_domains": [[2.0, None]],
            "horizons": [2.0],
            "photon_spheres": [3.0],
            "g_convention": "g_rr_inverse",
            "B_convention": "g_rr",
        },
        source_format="li-dm-bh",
        original_keys=("r", "f", "g", "B", "params"),
        raw_B=np.where(C != 0.0, 1.0 / C, np.nan),
    )
    payload = CanonicalPayload(r=r, A=A, C=C, metadata=metadata)
    with pytest.raises(ValueError, match="non-positive"):
        TabulatedMetric(payload=payload, static_domain=(2.0, math.inf))


def test_validation_gate_warns_on_photon_sphere_residual_above_threshold():
    """Plan decision 8 soft list: ``r_ph`` disagreement vs tabulated ``rA'-2A=0``
    zeros should soft-warn when residual exceeds ``photon_sphere_residual_warn``.

    We build a Schwarzschild-like ``A(r) = 1 - 2/r`` whose photon sphere is
    ``r=3`` analytically, and then lie about it in metadata to force a residual
    well above ``1e-6``. ``horizons`` is intentionally empty so the horizon
    residual check (which is a *hard* fail) does not short-circuit the soft
    check we want to exercise.
    """
    from spherical_raytracing import ValidationGate

    r = np.linspace(2.05, 10.0, 200)
    A = 1.0 - 2.0 / r
    C = A.copy()
    payload = _make_li_payload(
        r, A, C,
        params_overrides={
            "horizons": [],
            "photon_spheres": [3.5],  # off by ~0.5 from the true r_ph=3
            "static_domains": [[2.05, None]],
        },
    )
    with pytest.warns(UserWarning, match="photon_spheres"):
        ValidationGate().run(payload)


def test_validation_gate_warns_on_finite_r_max_with_unbounded_domain(tmp_path):
    """A Schwarzschild-like table (``static_domains=[[r_h, inf]]``) but bounded
    grid (``r_max=500``) should trigger the "r_max finite, prefer
    FiniteStaticObserver" soft warning (plan decision 8 soft list, last item).

    ``horizons`` / ``photon_spheres`` are left empty so unrelated soft / hard
    residual checks do not interfere with the warning we want to assert.
    """
    from spherical_raytracing import ValidationGate

    r = np.linspace(2.05, 500.0, 300)
    A = 1.0 - 2.0 / r
    C = A.copy()
    payload = _make_li_payload(
        r, A, C,
        params_overrides={
            "horizons": [],
            "photon_spheres": [],
            "static_domains": [[2.05, None]],
        },
    )
    with pytest.warns(UserWarning, match="r_max=.* finite"):
        ValidationGate().run(payload)


# =====================================================================
# Codex review pass: zero-anchor / r > r_grid_max / inferred roots
# =====================================================================
def _build_standard_rab_schwarzschild_payload(r_lo=2.1, r_hi=50.0, num=400):
    """Standard ``r/A/B`` Schwarzschild payload (no metadata horizons/photon_spheres)."""
    from spherical_raytracing.numerical_metrics import CanonicalPayload, LiDMBHMetadata

    r = np.linspace(r_lo, r_hi, num)
    A = 1.0 - 2.0 / r
    C = A.copy()
    metadata = LiDMBHMetadata.from_params(
        params=None,
        source_format="standard-rab",
        original_keys=("r", "A", "B"),
        raw_B=None,
    )
    return CanonicalPayload(r=r, A=A, C=C, metadata=metadata)


def test_tabulated_metric_does_not_zero_anchor_non_horizon_bounds():
    """A standard ``r/A/B`` payload with no metadata horizons must not have
    ``(r_lo, 0)`` / ``(r_hi, 0)`` anchors -- those would inject fake horizons
    and distort interpolation near the support edges. ``A`` near the edges
    should match the analytical value, not be pulled toward zero.
    """
    from spherical_raytracing import TabulatedMetric

    payload = _build_standard_rab_schwarzschild_payload(r_lo=2.1, r_hi=50.0, num=400)
    metric = TabulatedMetric(payload=payload, static_domain=(2.1, 50.0))

    for r_query in (2.105, 2.2, 5.0, 25.0, 49.5, 49.9):
        analytical = 1.0 - 2.0 / r_query
        assert math.isclose(metric.A(r_query), analytical, rel_tol=1e-3, abs_tol=1e-3), (
            f"A({r_query})={metric.A(r_query)} vs analytical {analytical}: "
            "edge values are being pulled toward zero by a spurious anchor"
        )


def test_tabulated_metric_radial_bounds_override_does_not_zero_anchor():
    """When ``StaticDomainSelector(radial_bounds=...)`` crops *inside* a static
    patch, the user-supplied bounds are not horizons; they must not be zero-
    anchored either.
    """
    from spherical_raytracing.numerical_metrics import (
        CanonicalPayload,
        LiDMBHMetadata,
        TabulatedMetric,
    )

    r = np.linspace(2.05, 30.0, 600)
    A = 1.0 - 2.0 / r
    C = A.copy()
    metadata = LiDMBHMetadata.from_params(
        params={
            "static_domains": [[2.0, None]],
            "horizons": [2.0],
            "photon_spheres": [3.0],
            "g_convention": "g_rr_inverse",
            "B_convention": "g_rr",
        },
        source_format="li-dm-bh",
        original_keys=("r", "f", "g", "B", "params"),
        raw_B=1.0 / C,
    )
    payload = CanonicalPayload(r=r, A=A, C=C, metadata=metadata)
    # 5.0 and 10.0 are not horizons (only r=2.0 is)
    metric = TabulatedMetric(payload=payload, static_domain=(5.0, 10.0))

    for r_query in (5.05, 5.5, 7.0, 9.5, 9.95):
        analytical = 1.0 - 2.0 / r_query
        assert math.isclose(metric.A(r_query), analytical, rel_tol=1e-3), (
            f"A({r_query})={metric.A(r_query)} distorted by anchor: expected {analytical}"
        )


def test_tabulated_metric_a_b_raise_when_r_above_grid_max():
    """For an unbounded metadata domain (``r_hi=inf``) backed by a finite grid,
    queries above ``r_grid_max`` must hard-fail in ``A`` / ``B`` / ``dA_dr`` /
    ``dB_dr``. Without this guard, PCHIP returns NaN silently and downstream
    callers (FiniteStaticObserver, the b_max guard) propagate NaN into the
    solver, which then fails in confusing ways.
    """
    from spherical_raytracing import StaticDomainSelector, TabulatedMetric, load_metric_npz

    _skip_if_li_data_missing()

    payload = load_metric_npz(LI_DATA_DIR / "hernquist_L0.npz")
    r_lo, r_hi = StaticDomainSelector().choose(payload)
    assert not math.isfinite(r_hi)  # unbounded metadata
    metric = TabulatedMetric(payload=payload, static_domain=(r_lo, r_hi))
    r_grid_max = float(np.max(payload.r))

    r_beyond = r_grid_max + 1.0
    for method_name in ("A", "B", "dA_dr", "dB_dr"):
        with pytest.raises(ValueError, match="above tabulated grid"):
            getattr(metric, method_name)(r_beyond)


def test_tabulated_metric_infers_photon_spheres_when_metadata_absent():
    """A standard ``r/A/B`` Schwarzschild table has no metadata photon spheres;
    ``photon_spheres()`` must fall back to numerically inferring the root of
    ``r * A'(r) - 2 * A(r) = 0`` so ``critical_curves()`` and ``b_crit`` remain
    available on the documented standard-table path.
    """
    from spherical_raytracing import TabulatedMetric

    payload = _build_standard_rab_schwarzschild_payload(r_lo=2.1, r_hi=50.0, num=400)
    metric = TabulatedMetric(payload=payload, static_domain=(2.1, 50.0))

    ps = metric.photon_spheres()
    assert ps, "photon_spheres() returned [] -- inference failed"
    assert len(ps) == 1
    assert math.isclose(ps[0], 3.0, rel_tol=1e-3, abs_tol=1e-3), (
        f"inferred r_ph={ps[0]} vs analytical 3.0"
    )

    curves = metric.critical_curves()
    assert curves, "critical_curves() returned [] -- standard r/A/B path broken"
    b_crit = curves[0].b_crit
    b_crit_analytical = 3.0 * math.sqrt(3.0)
    assert math.isclose(b_crit, b_crit_analytical, rel_tol=1e-3), (
        f"b_crit_inferred={b_crit} vs analytical {b_crit_analytical}"
    )


def test_tabulated_metric_horizons_inference_handles_no_root_in_grid():
    """When the grid is entirely above the horizon (e.g. standard Schwarzschild
    table on ``[2.1, 50]``), no ``A=0`` root exists inside the support and
    ``horizons()`` should return ``[]`` (not crash, not invent a fake root)."""
    from spherical_raytracing import TabulatedMetric

    payload = _build_standard_rab_schwarzschild_payload(r_lo=2.1, r_hi=50.0, num=400)
    metric = TabulatedMetric(payload=payload, static_domain=(2.1, 50.0))
    assert metric.horizons() == []


def test_tabulated_metric_metadata_takes_priority_over_inference():
    """Metadata-provided ``horizons`` / ``photon_spheres`` must win when present;
    we should not re-run inference and overwrite trusted upstream roots."""
    from spherical_raytracing.numerical_metrics import (
        CanonicalPayload,
        LiDMBHMetadata,
        TabulatedMetric,
    )

    r = np.linspace(2.05, 50.0, 400)
    A = 1.0 - 2.0 / r
    C = A.copy()
    metadata = LiDMBHMetadata.from_params(
        params={
            "static_domains": [[2.0, None]],
            "horizons": [2.0],
            "photon_spheres": [3.0],
            "g_convention": "g_rr_inverse",
            "B_convention": "g_rr",
        },
        source_format="li-dm-bh",
        original_keys=("r", "f", "g", "B", "params"),
        raw_B=1.0 / C,
    )
    payload = CanonicalPayload(r=r, A=A, C=C, metadata=metadata)
    metric = TabulatedMetric(payload=payload, static_domain=(2.0, math.inf))

    assert metric.horizons() == [2.0]
    assert metric.photon_spheres() == [3.0]


def test_tabulated_metric_schwarzschild_end_to_end_via_standard_rab():
    """End-to-end smoke for the documented standard ``r/A/B`` path: a
    Schwarzschild table without metadata must still trace via inferred
    photon spheres / b_crit (Codex P3 acceptance)."""
    from spherical_raytracing import (
        FiniteStaticObserver,
        QuadTransferSolver,
        TabulatedMetric,
    )

    payload = _build_standard_rab_schwarzschild_payload(r_lo=2.05, r_hi=200.0, num=4000)
    metric = TabulatedMetric(payload=payload, static_domain=(2.05, 200.0))
    crits = metric.critical_curves()
    assert crits, "no critical_curves() inferred for standard r/A/B Schwarzschild"
    b_crit = crits[0].b_crit

    observer = FiniteStaticObserver(r_obs=80.0, metric=metric)
    solver = QuadTransferSolver(metric=metric, observer=observer)
    result = solver.trace_b(b_crit * 1.05)
    assert result.segments, "trace produced no segments"
    last = result.segments[-1]
    assert math.isfinite(last.phi_end) and last.phi_end > 0
