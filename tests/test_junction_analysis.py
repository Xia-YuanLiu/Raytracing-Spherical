import math

import pytest

from spherical_raytracing.junction_analysis import (
    AtlasCase,
    AtlasParameters,
    admit_parameters,
    build_junction,
    build_metrics,
    classify_photon_spheres,
    detect_peaks,
    diagnose_profile,
    PhysicalFilterConfig,
    augment_case_tags_from_profile,
    diagnostic_signal_score,
    paper_style_emissivity_factory,
    paper_style_inner_edge,
    select_representative_cases,
)
from spherical_raytracing.metrics import ReissnerNordstromDeSitterMetric, ReissnerNordstromMetric, SchwarzschildMetric
from spherical_raytracing.policies import SolverOptions


def test_physical_filter_rejects_a_nonstatic_shell():
    result = admit_parameters(
        AtlasParameters(
            family="schwarzschild",
            m_minus=0.1,
            m_plus=1.0,
            shell_radius=1.0,
            observer_radius=50.0,
        )
    )

    assert result.admitted is False
    assert "outer_shell_radius_nonstatic" in result.rejection_reasons


def test_physical_filter_rejects_observer_inside_shell():
    result = admit_parameters(
        AtlasParameters(
            family="schwarzschild",
            m_minus=0.1,
            m_plus=1.0,
            shell_radius=5.0,
            observer_radius=4.0,
        )
    )

    assert result.admitted is False
    assert "observer_inside_shell" in result.rejection_reasons


def test_physical_filter_allows_negative_sigma_but_records_warning():
    result = admit_parameters(
        AtlasParameters(
            family="schwarzschild",
            m_minus=1.0,
            m_plus=0.1,
            shell_radius=5.0,
            observer_radius=50.0,
        )
    )

    assert result.admitted is True
    assert result.surface_energy_density is not None
    assert result.surface_energy_density < 0.0
    assert "negative_surface_energy_density" in result.warnings


def test_physical_filter_records_q_and_lambda_jumps_without_rejecting():
    result = admit_parameters(
        AtlasParameters(
            family="rnds",
            m_minus=0.8,
            m_plus=1.0,
            q_minus=0.1,
            q_plus=0.2,
            lambda_minus=0.0,
            lambda_plus=0.005,
            shell_radius=5.0,
            observer_radius=6.0,
        )
    )

    assert result.admitted is True
    assert result.q_jump == pytest.approx(0.1)
    assert result.lambda_jump == pytest.approx(0.005)
    assert "charge_mismatch_surface_source" in result.warnings
    assert "lambda_mismatch_vacuum_discontinuity" in result.warnings


def test_physical_filter_rejects_overextremal_rn_nakedness():
    result = admit_parameters(
        AtlasParameters(
            family="rn",
            m_minus=0.8,
            m_plus=1.0,
            q_minus=0.2,
            q_plus=1.2,
            shell_radius=5.0,
            observer_radius=50.0,
        )
    )

    assert result.admitted is False
    assert "outer_naked_singularity" in result.rejection_reasons


def test_physical_filter_rejects_rnds_observer_outside_clean_patch():
    metric = ReissnerNordstromDeSitterMetric(mass=1.0, charge=0.2, cosmological_constant=0.01)
    cosmological_horizon = metric.valid_radial_domain()[1]

    result = admit_parameters(
        AtlasParameters(
            family="rnds",
            m_minus=0.8,
            m_plus=1.0,
            q_minus=0.2,
            q_plus=0.2,
            lambda_minus=0.01,
            lambda_plus=0.01,
            shell_radius=5.0,
            observer_radius=cosmological_horizon * 1.01,
        )
    )

    assert result.admitted is False
    assert "outer_rnds_observer_outside_clean_static_patch" in result.rejection_reasons


def test_physical_filter_can_keep_rnds_clean_patch_warning_nonrejecting_when_disabled(monkeypatch):
    from spherical_raytracing.junctions import StaticJunctionSpacetime

    class FakeDiagnostics:
        warnings = ("outer_rnds_no_clean_static_patch",)
        surface_energy_density = 0.0
        surface_pressure = 1.0

    def fake_physics_diagnostics(self, r_obs=None):
        return FakeDiagnostics()

    monkeypatch.setattr(StaticJunctionSpacetime, "physics_diagnostics", fake_physics_diagnostics)

    result = admit_parameters(
        AtlasParameters(
            family="rnds",
            m_minus=0.8,
            m_plus=1.0,
            q_minus=0.2,
            q_plus=0.2,
            lambda_minus=0.01,
            lambda_plus=0.01,
            shell_radius=5.0,
            observer_radius=6.0,
        ),
        config=PhysicalFilterConfig(require_clean_rnds_patch=False),
    )

    assert result.admitted is True
    assert "outer_rnds_no_clean_static_patch" in result.warnings
    assert "outer_rnds_no_clean_static_patch" not in result.rejection_reasons


def test_physical_filter_rejects_rnds_naked_singularity():
    result = admit_parameters(
        AtlasParameters(
            family="rnds",
            m_minus=1.0,
            m_plus=1.0,
            q_minus=1.2,
            q_plus=1.2,
            lambda_minus=0.01,
            lambda_plus=0.01,
            shell_radius=5.0,
            observer_radius=6.0,
        )
    )

    assert result.admitted is False
    assert "outer_naked_singularity" in result.rejection_reasons


def test_physical_filter_rejects_nonfinite_surface_pressure(monkeypatch):
    from spherical_raytracing.junctions import StaticJunctionSpacetime

    class FakeDiagnostics:
        warnings = ()
        surface_energy_density = 0.0
        surface_pressure = math.inf

    def fake_physics_diagnostics(self, r_obs=None):
        return FakeDiagnostics()

    monkeypatch.setattr(StaticJunctionSpacetime, "physics_diagnostics", fake_physics_diagnostics)

    result = admit_parameters(
        AtlasParameters(
            family="schwarzschild",
            m_minus=1.0,
            m_plus=1.0,
            shell_radius=5.0,
            observer_radius=50.0,
        )
    )

    assert result.admitted is False
    assert "non_finite_surface_pressure" in result.rejection_reasons


def test_physical_filter_admits_rnds_lambda_zero_as_rn_boundary():
    result = admit_parameters(
        AtlasParameters(
            family="rnds",
            m_minus=0.8,
            m_plus=1.0,
            q_minus=0.2,
            q_plus=0.2,
            lambda_minus=0.0,
            lambda_plus=0.0,
            shell_radius=5.0,
            observer_radius=50.0,
        )
    )

    assert result.admitted is True
    assert result.rejection_reasons == ()


def test_build_metrics_and_build_junction_wire_same_family_and_horizon_buffer():
    params = AtlasParameters(
        family="rn",
        m_minus=0.8,
        m_plus=1.0,
        q_minus=0.1,
        q_plus=0.2,
        shell_radius=5.0,
        observer_radius=50.0,
    )

    inner, outer = build_metrics(params)
    junction, observer = build_junction(params, options=SolverOptions(horizon_buffer=0.123))

    assert isinstance(inner, ReissnerNordstromMetric)
    assert isinstance(outer, ReissnerNordstromMetric)
    assert junction.inner_metric.region == "inner"
    assert junction.outer_metric.region == "outer"
    assert junction.options.horizon_buffer == pytest.approx(0.123)
    assert observer.r_obs == pytest.approx(50.0)
    assert isinstance(observer.metric, ReissnerNordstromMetric)


def test_paper_style_inner_edge_behavior_for_schwarzschild_rn_and_rnds():
    schwarzschild = SchwarzschildMetric(mass=0.1)
    rn = ReissnerNordstromMetric(mass=0.1, charge=0.02)
    rnds = ReissnerNordstromDeSitterMetric(mass=1.0, charge=0.2, cosmological_constant=0.01)

    assert paper_style_inner_edge(schwarzschild, shell_radius=3.1) == pytest.approx(0.2)
    assert paper_style_inner_edge(rn, shell_radius=3.1) == pytest.approx(rn.horizons()[-1])
    assert paper_style_inner_edge(rnds, shell_radius=5.0) == pytest.approx(rnds.valid_radial_domain()[0])


def test_paper_style_inner_edge_rejects_horizonless_naked_metrics():
    overextremal_rn = ReissnerNordstromMetric(mass=1.0, charge=1.2)
    naked_rnds = ReissnerNordstromDeSitterMetric(mass=1.0, charge=1.2, cosmological_constant=0.01)

    with pytest.raises(ValueError, match="paper emissivity requires"):
        paper_style_inner_edge(overextremal_rn, shell_radius=5.0)
    with pytest.raises(ValueError, match="paper emissivity requires"):
        paper_style_inner_edge(naked_rnds, shell_radius=5.0)


def test_paper_style_emissivity_factory_uses_the_family_specific_inner_edge():
    rn = ReissnerNordstromMetric(mass=0.1, charge=0.02)
    edge = paper_style_inner_edge(rn, shell_radius=3.1)
    emissivity = paper_style_emissivity_factory(rn, shell_radius=3.1)

    assert emissivity(edge * 0.999) == 0.0
    assert emissivity(edge) == pytest.approx(1.0)
    assert emissivity(edge + 1.0) == pytest.approx(1.0 / 8.0)


def test_classify_photon_spheres_for_schwarzschild_reference_regions():
    inner_only = classify_photon_spheres(
        AtlasParameters(family="schwarzschild", m_minus=0.1, m_plus=1.0, shell_radius=3.1, observer_radius=50.0)
    )
    double = classify_photon_spheres(
        AtlasParameters(family="schwarzschild", m_minus=0.1, m_plus=1.0, shell_radius=2.9, observer_radius=50.0)
    )

    assert inner_only.classification == "inner_only"
    assert inner_only.visible_regions == ("inner",)
    assert double.classification == "double"
    assert double.visible_regions == ("inner", "outer")


def test_classify_photon_spheres_reports_outer_only_and_none_regions():
    outer_only = classify_photon_spheres(
        AtlasParameters(family="rn", m_minus=0.8, m_plus=1.0, q_minus=1.2, q_plus=0.2, shell_radius=1.0)
    )
    none = classify_photon_spheres(
        AtlasParameters(family="rn", m_minus=0.8, m_plus=1.0, q_minus=1.2, q_plus=1.2, shell_radius=10.0)
    )

    assert outer_only.classification == "outer_only"
    assert outer_only.visible_regions == ("outer",)
    assert none.classification == "none"
    assert none.visible_regions == ()


def test_detect_peaks_finds_prominent_local_maxima():
    peaks = detect_peaks([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 3.0, 1.0, 4.0, 0.0], min_prominence=1.0)

    assert [peak.index for peak in peaks] == [1, 3]
    assert peaks[0].x == pytest.approx(1.0)
    assert peaks[1].y == pytest.approx(4.0)


def test_detect_peaks_rejects_mismatched_coordinate_lengths():
    with pytest.raises(ValueError, match="same length"):
        detect_peaks([0.0, 1.0], [0.0])


def test_diagnose_profile_reports_cusp_and_v_shape_orders_from_rows():
    rows = [
        {"b_plus": 1.0, "shell_crossing_count": 1, "intersections": [{"m": 1, "r": 4.9, "g": 0.5}]},
        {"b_plus": 2.0, "shell_crossing_count": 1, "intersections": [{"m": 1, "r": 5.0, "g": 0.7}]},
        {"b_plus": 3.0, "shell_crossing_count": 1, "intersections": [{"m": 1, "r": 4.8, "g": 0.55}]},
    ]

    diagnostics = diagnose_profile(rows, shell_radius=5.0)

    assert diagnostics.peak_count == 1
    assert diagnostics.redshift_cusp_orders == (1,)
    assert diagnostics.v_shape_orders == (1,)
    assert diagnostics.shell_crossing_count_max == 1


def test_diagnose_profile_handles_empty_and_no_intersection_rows():
    empty = diagnose_profile([], shell_radius=5.0)
    no_intersections = diagnose_profile(
        [
            {"b_plus": 1.0, "intensity": 0.0, "shell_crossing_count": 0, "intersections": []},
            {"b_plus": 2.0, "intensity": 0.0, "shell_crossing_count": 0, "intersections": []},
        ],
        shell_radius=5.0,
    )

    assert empty.peak_count == 0
    assert empty.redshift_cusp_orders == ()
    assert empty.v_shape_orders == ()
    assert empty.shell_crossing_count_max == 0
    assert no_intersections.peak_count == 0
    assert no_intersections.redshift_cusp_orders == ()
    assert no_intersections.v_shape_orders == ()
    assert no_intersections.shell_crossing_count_max == 0


def test_select_representative_cases_is_deterministic_and_preserves_all_tags():
    candidates = [
        AtlasCase(
            "b",
            "rn",
            AtlasParameters(family="rn", m_minus=0.8, m_plus=1.0, q_plus=0.2),
            ("high_charge",),
            {"high_charge": 0.30, "near_schwarzschild": 0.80},
            0.9,
        ),
        AtlasCase(
            "a",
            "rn",
            AtlasParameters(family="rn", m_minus=0.8, m_plus=1.0, q_plus=0.2),
            ("high_charge", "near_schwarzschild"),
            {"high_charge": 0.10, "near_schwarzschild": 0.05},
            0.4,
        ),
        AtlasCase(
            "c",
            "rn",
            AtlasParameters(family="rn", m_minus=0.8, m_plus=1.0, q_plus=0.3),
            ("near_schwarzschild",),
            {"high_charge": 0.60, "near_schwarzschild": 0.40},
            0.2,
        ),
    ]

    first = select_representative_cases(candidates, per_category_limit=1)
    second = select_representative_cases(reversed(candidates), per_category_limit=1)

    assert first == second
    assert [case.case_id for case in first] == ["a"]
    assert first[0].category_tags == ("high_charge", "near_schwarzschild")


def test_select_representative_cases_records_empty_categories():
    selected, empty = select_representative_cases(
        [
            AtlasCase(
                "a",
                "rn",
                AtlasParameters(family="rn", m_minus=0.8, m_plus=1.0),
                ("inner_only",),
                {"inner_only": 0.0},
                0.0,
            )
        ],
        per_category_limit=1,
        required_categories=("inner_only", "outer_only"),
        return_empty=True,
    )

    assert [case.case_id for case in selected] == ["a"]
    assert empty == ("outer_only",)


def test_select_representative_cases_merges_duplicate_tags_and_backfills_category_quota():
    shared_parameters = AtlasParameters(family="rn", m_minus=0.8, m_plus=1.0, q_plus=0.3)
    candidates = [
        AtlasCase(
            "shared",
            "rn",
            shared_parameters,
            ("high_charge",),
            {"high_charge": 0.0, "near_schwarzschild": 0.1},
            1.0,
        ),
        AtlasCase(
            "shared",
            "rn",
            shared_parameters,
            ("near_schwarzschild",),
            {"high_charge": 0.0, "near_schwarzschild": 0.0},
            2.0,
        ),
        AtlasCase(
            "alternate",
            "rn",
            AtlasParameters(family="rn", m_minus=0.9, m_plus=1.0, q_plus=0.0),
            ("near_schwarzschild",),
            {"near_schwarzschild": 0.2},
            0.5,
        ),
    ]

    selected = select_representative_cases(
        candidates,
        per_category_limit=2,
        required_categories=("high_charge", "near_schwarzschild"),
    )

    by_id = {case.case_id: case for case in selected}
    assert set(by_id) == {"alternate", "shared"}
    assert by_id["shared"].category_tags == ("high_charge", "near_schwarzschild")
    assert by_id["shared"].signal_score == pytest.approx(2.0)


def test_select_representative_cases_rejects_nonpositive_category_limits():
    candidate = AtlasCase(
        "a",
        "rn",
        AtlasParameters(family="rn", m_minus=0.8, m_plus=1.0),
        ("inner_only",),
        {"inner_only": 0.0},
        0.0,
    )

    with pytest.raises(ValueError, match="per_category_limit"):
        select_representative_cases([candidate], per_category_limit=0)
    with pytest.raises(ValueError, match="per_category_limit"):
        select_representative_cases([candidate], per_category_limit=-1)


def test_select_representative_cases_rejects_duplicate_case_id_with_inconsistent_family():
    parameters = AtlasParameters(family="rn", m_minus=0.8, m_plus=1.0)
    candidates = [
        AtlasCase("shared", "rn", parameters, ("inner_only",), {"inner_only": 0.0}, 0.0),
        AtlasCase("shared", "rnds", parameters, ("outer_only",), {"outer_only": 0.0}, 0.0),
    ]

    with pytest.raises(ValueError, match="multiple families"):
        select_representative_cases(
            candidates,
            per_category_limit=1,
            required_categories=("inner_only", "outer_only"),
        )


def test_diagnostic_signal_score_counts_sequence_warnings_and_weighted_jumps():
    score = diagnostic_signal_score(
        {
            "warnings": ("negative_surface_energy_density", "charge_mismatch_surface_source"),
            "q_jump": 0.2,
            "lambda_jump": 0.01,
            "photon_sphere_classification": "double",
        }
    )

    assert score == pytest.approx(2.0 + 2.0 + 10.0 * 0.2 + 200.0 * 0.01)


def test_augment_case_tags_from_profile_adds_profile_derived_tags():
    case = AtlasCase(
        "case",
        "rnds",
        AtlasParameters(
            family="rnds",
            m_minus=0.8,
            m_plus=1.0,
            lambda_plus=0.01,
            shell_radius=5.0,
        ),
        ("double",),
        {"double": 0.0},
        1.0,
    )
    profile = {
        "profile_diagnostics": {"peak_count": 1, "redshift_cusp_orders": [1]},
        "samples": [
            {"intersections": [{"r": 5.0005}]},
        ],
    }

    augmented = augment_case_tags_from_profile(case, profile)

    assert "double_sphere_single_ring" in augmented.category_tags
    assert "shell_near_outer_photon_sphere" in augmented.category_tags
    assert "rnds_observer_near_cosmological_patch" in augmented.category_tags
