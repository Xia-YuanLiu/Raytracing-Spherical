import math

import pytest

from spherical_raytracing.diagnostics import EventType, RayDiagnostics, RaySegment
from spherical_raytracing.junctions import (
    JunctionRayResult,
    SegmentConstants,
    StaticJunctionSpacetime,
    StaticShell,
    junction_redshift_factor,
    match_static_shell,
)
from spherical_raytracing.metrics import ReissnerNordstromDeSitterMetric, ReissnerNordstromMetric, SchwarzschildMetric
from spherical_raytracing.observers import FiniteStaticObserver
from spherical_raytracing.transfer import IntersectionRecord


def test_shell_matching_preserves_l_and_updates_e_and_b():
    inner = SchwarzschildMetric(mass=0.1, region="inner")
    outer = SchwarzschildMetric(mass=1.0, region="outer")
    junction = StaticJunctionSpacetime(inner_metric=inner, outer_metric=outer, shell=StaticShell(radius=5.0))

    matched = match_static_shell(junction, from_region="outer", to_region="inner", E=1.0, L=4.0, b=4.0)

    expected_ratio = math.sqrt(inner.A(5.0) / outer.A(5.0))
    assert matched.region == "inner"
    assert matched.E == pytest.approx(expected_ratio)
    assert matched.L == pytest.approx(4.0)
    assert matched.b == pytest.approx(4.0 / expected_ratio)


def test_junction_diagnostics_warn_for_negative_sigma_and_mismatched_charge_and_lambda():
    inner = ReissnerNordstromDeSitterMetric(
        mass=0.8,
        charge=0.1,
        cosmological_constant=0.02,
        region="inner",
    )
    outer = ReissnerNordstromDeSitterMetric(
        mass=1.0,
        charge=0.4,
        cosmological_constant=0.01,
        region="outer",
    )
    junction = StaticJunctionSpacetime(inner_metric=inner, outer_metric=outer, shell=StaticShell(radius=5.0))

    diagnostics = junction.physics_diagnostics(r_obs=6.0)

    assert "charge_mismatch_surface_source" in diagnostics.warnings
    assert "lambda_mismatch_vacuum_discontinuity" in diagnostics.warnings
    assert diagnostics.surface_energy_density == pytest.approx(
        (math.sqrt(inner.A(5.0)) - math.sqrt(outer.A(5.0))) / (4.0 * math.pi * 5.0)
    )


def test_junction_diagnostics_warn_for_observer_near_shell():
    inner = SchwarzschildMetric(mass=0.1, region="inner")
    outer = SchwarzschildMetric(mass=1.0, region="outer")
    junction = StaticJunctionSpacetime(inner_metric=inner, outer_metric=outer, shell=StaticShell(radius=5.0))

    diagnostics = junction.physics_diagnostics(r_obs=5.0 + junction.options.horizon_buffer)

    assert "observer_near_shell" in diagnostics.warnings


def test_junction_diagnostics_warn_for_overcharged_rnds_without_black_hole_horizon():
    inner = ReissnerNordstromDeSitterMetric(
        mass=1.0,
        charge=1.2,
        cosmological_constant=0.01,
        region="inner",
    )
    outer = ReissnerNordstromDeSitterMetric(
        mass=1.0,
        charge=1.2,
        cosmological_constant=0.01,
        region="outer",
    )
    junction = StaticJunctionSpacetime(inner_metric=inner, outer_metric=outer, shell=StaticShell(radius=5.0))

    diagnostics = junction.physics_diagnostics(r_obs=6.0)

    assert "inner_naked_singularity" in diagnostics.warnings
    assert "outer_naked_singularity" in diagnostics.warnings


def test_rnds_only_cosmological_horizon_is_naked_with_no_clean_patch_warning():
    inner = ReissnerNordstromDeSitterMetric(
        mass=1.0,
        charge=1.2,
        cosmological_constant=0.01,
        region="inner",
    )
    outer = ReissnerNordstromDeSitterMetric(
        mass=1.0,
        charge=1.2,
        cosmological_constant=0.01,
        region="outer",
    )
    junction = StaticJunctionSpacetime(inner_metric=inner, outer_metric=outer, shell=StaticShell(radius=5.0))

    assert len(inner.horizons()) == 1
    diagnostics = junction.physics_diagnostics(r_obs=6.0)

    assert "inner_naked_singularity" in diagnostics.warnings
    assert "outer_naked_singularity" in diagnostics.warnings
    assert "inner_rnds_no_clean_static_patch" in diagnostics.warnings
    assert "outer_rnds_no_clean_static_patch" in diagnostics.warnings


def test_rnds_lambda_zero_boundary_does_not_emit_clean_patch_warning():
    inner = ReissnerNordstromDeSitterMetric(
        mass=0.8,
        charge=0.2,
        cosmological_constant=0.0,
        region="inner",
    )
    outer = ReissnerNordstromDeSitterMetric(
        mass=1.0,
        charge=0.2,
        cosmological_constant=0.0,
        region="outer",
    )
    junction = StaticJunctionSpacetime(inner_metric=inner, outer_metric=outer, shell=StaticShell(radius=5.0))

    diagnostics = junction.physics_diagnostics(r_obs=50.0)

    assert "inner_rnds_no_clean_static_patch" not in diagnostics.warnings
    assert "outer_rnds_no_clean_static_patch" not in diagnostics.warnings


def test_rnds_observer_outside_black_hole_to_cosmological_patch_is_not_naked_but_warns():
    inner = ReissnerNordstromDeSitterMetric(
        mass=1.0,
        charge=0.5,
        cosmological_constant=0.01,
        region="inner",
    )
    outer = ReissnerNordstromDeSitterMetric(
        mass=1.0,
        charge=0.5,
        cosmological_constant=0.01,
        region="outer",
    )
    horizons = outer.horizons()
    junction = StaticJunctionSpacetime(
        inner_metric=inner,
        outer_metric=outer,
        shell=StaticShell(radius=0.5 * (horizons[-2] + horizons[-1])),
    )

    diagnostics = junction.physics_diagnostics(r_obs=horizons[-1] + 1.0)

    assert "outer_naked_singularity" not in diagnostics.warnings
    assert "outer_rnds_no_clean_static_patch" in diagnostics.warnings


def test_static_junction_rejects_mixed_metric_families():
    with pytest.raises(ValueError, match="same-family"):
        StaticJunctionSpacetime(
            inner_metric=SchwarzschildMetric(mass=0.1, region="inner"),
            outer_metric=ReissnerNordstromMetric(mass=1.0, charge=0.2, region="outer"),
            shell=StaticShell(radius=5.0),
        )


def test_junction_result_b_alias_is_b_plus():
    diagnostics = RayDiagnostics(
        estimated_error=0.0,
        near_critical=False,
        critical_curve_id=None,
        hit_horizon=False,
        hit_inner_boundary=False,
        turning_point_count=0,
        max_phi_reached=False,
        termination_reason="escape",
        residuals={},
    )
    result = JunctionRayResult(
        input_coordinate=0.1,
        coordinate_name="alpha",
        b_plus=3.0,
        segments=[],
        segment_constants=[],
        events=[],
        shell_crossings=[],
        physics_warnings=[],
        diagnostics=diagnostics,
        backend_name="unit",
    )

    assert result.b == 3.0
    assert result.termination_reason == "escape"


def test_shell_crossing_event_type_is_distinct():
    assert EventType.SHELL_CROSSING.value == "shell_crossing"
    assert EventType.SHELL_CROSSING is not EventType.REGION_TRANSITION


def test_forward_time_redshift_inverts_backward_shell_crossing_ratio():
    inner = SchwarzschildMetric(mass=0.1, region="inner")
    outer = SchwarzschildMetric(mass=1.0, region="outer")
    junction = StaticJunctionSpacetime(inner_metric=inner, outer_metric=outer, shell=StaticShell(radius=5.0))
    constants_outer = SegmentConstants(region="outer", E=1.0, L=4.0, b=4.0)
    constants_inner = match_static_shell(junction, from_region="outer", to_region="inner", E=1.0, L=4.0, b=4.0)
    segment = RaySegment(
        region="inner",
        radial_direction="inward",
        phi_start=0.1,
        phi_end=2.0,
        u_start=0.2,
        u_end=0.3,
        endpoint_event=EventType.MAX_PHI,
        evaluator=lambda phi: 0.25,
    )
    crossing = junction.shell_crossing_record(
        phi=0.1,
        radial_direction="inward",
        before=constants_outer,
        after=constants_inner,
    )
    diagnostics = RayDiagnostics(
        estimated_error=0.0,
        near_critical=False,
        critical_curve_id=None,
        hit_horizon=False,
        hit_inner_boundary=False,
        turning_point_count=0,
        max_phi_reached=True,
        termination_reason="max_phi",
        residuals={},
    )
    result = JunctionRayResult(
        input_coordinate=0.1,
        coordinate_name="alpha",
        b_plus=4.0,
        segments=[segment],
        segment_constants=[constants_inner],
        events=[],
        shell_crossings=[crossing],
        physics_warnings=[],
        diagnostics=diagnostics,
        backend_name="unit",
    )
    intersection = IntersectionRecord(
        m=1,
        r=4.0,
        phi=1.0,
        region="inner",
        ray_segment=segment,
        path_class="direct",
    )
    observer = FiniteStaticObserver(r_obs=50.0, metric=outer)

    g = junction_redshift_factor(intersection, result, observer, {"inner": inner, "outer": outer})

    assert g == pytest.approx(
        math.sqrt(inner.A(4.0) / outer.A(50.0)) * math.sqrt(outer.A(5.0) / inner.A(5.0))
    )


def test_shell_matching_residuals_validate_generic_optical_formula():
    inner = ReissnerNordstromMetric(mass=0.8, charge=0.2, region="inner")
    outer = ReissnerNordstromMetric(mass=1.0, charge=0.2, region="outer")
    junction = StaticJunctionSpacetime(inner_metric=inner, outer_metric=outer, shell=StaticShell(radius=5.0))

    matched = match_static_shell(junction, from_region="outer", to_region="inner", E=1.0, L=4.0, b=4.0)
    residuals = junction.shell_matching_residuals(
        from_region="outer",
        to_region="inner",
        before=SegmentConstants(region="outer", E=1.0, L=4.0, b=4.0),
        after=matched,
    )

    assert residuals["energy_ratio_abs"] < 1e-12
    assert residuals["angular_momentum_abs"] < 1e-12
    assert residuals["impact_parameter_abs"] < 1e-12


def test_shell_matching_rejects_inconsistent_impact_parameter():
    inner = SchwarzschildMetric(mass=0.1, region="inner")
    outer = SchwarzschildMetric(mass=1.0, region="outer")
    junction = StaticJunctionSpacetime(inner_metric=inner, outer_metric=outer, shell=StaticShell(radius=5.0))

    with pytest.raises(ValueError, match="b must equal L / E"):
        match_static_shell(junction, from_region="outer", to_region="inner", E=1.0, L=4.0, b=5.0)


def test_direct_static_observer_redshift_matches_junction_helper():
    inner = ReissnerNordstromMetric(mass=0.8, charge=0.2, region="inner")
    outer = ReissnerNordstromMetric(mass=1.0, charge=0.2, region="outer")
    junction = StaticJunctionSpacetime(inner_metric=inner, outer_metric=outer, shell=StaticShell(radius=5.0))
    before = SegmentConstants(region="outer", E=1.0, L=4.0, b=4.0)
    after = match_static_shell(junction, from_region="outer", to_region="inner", E=before.E, L=before.L, b=before.b)
    crossing = junction.shell_crossing_record(phi=0.1, radial_direction="inward", before=before, after=after)
    segment = RaySegment(
        region="inner",
        radial_direction="inward",
        phi_start=0.1,
        phi_end=2.0,
        u_start=0.2,
        u_end=0.3,
        endpoint_event=EventType.MAX_PHI,
        evaluator=lambda phi: 0.25,
    )
    diagnostics = RayDiagnostics(
        estimated_error=0.0,
        near_critical=False,
        critical_curve_id=None,
        hit_horizon=False,
        hit_inner_boundary=False,
        turning_point_count=0,
        max_phi_reached=True,
        termination_reason="max_phi",
        residuals={},
    )
    result = JunctionRayResult(
        input_coordinate=0.1,
        coordinate_name="alpha",
        b_plus=4.0,
        segments=[segment],
        segment_constants=[after],
        events=[],
        shell_crossings=[crossing],
        physics_warnings=[],
        diagnostics=diagnostics,
        backend_name="unit",
    )
    intersection = IntersectionRecord(
        m=1,
        r=4.0,
        phi=1.0,
        region="inner",
        ray_segment=segment,
        path_class="direct",
    )
    observer = FiniteStaticObserver(r_obs=50.0, metric=outer)

    helper_g = junction_redshift_factor(intersection, result, observer, {"inner": inner, "outer": outer})
    direct_g = (before.E / math.sqrt(outer.A(observer.r_obs))) / (after.E / math.sqrt(inner.A(intersection.r)))

    assert helper_g == pytest.approx(direct_g)
