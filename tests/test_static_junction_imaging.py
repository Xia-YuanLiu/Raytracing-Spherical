from dataclasses import dataclass

from spherical_raytracing.imaging import SamplingOptions, sample_radial_profile


@dataclass(frozen=True)
class FakeObserved:
    total: float
    path_classes: list[str]
    diagnostics: dict[str, float | int | str]


def test_sampler_refines_on_shell_crossing_count_change():
    def intensity_fn(x: float) -> FakeObserved:
        count = 0 if x < 0.5 else 1
        return FakeObserved(
            total=1.0,
            path_classes=["direct"],
            diagnostics={"termination_reason": "escape", "shell_crossing_count": count},
        )

    profile = sample_radial_profile(
        0.0,
        1.0,
        intensity_fn,
        critical_points=[],
        options=SamplingOptions(initial_count=3, max_refine=1),
    )

    assert profile.diagnostics["refine_rounds"] == 1
    assert profile.diagnostics["sample_count"] > 3
    assert 0.25 in profile.coordinates
    assert 0.75 not in profile.coordinates
    assert profile.diagnostics["shell_crossing_change_intervals"] == 1
    assert profile.diagnostics["backend_disagreement_intervals"] == 0
    assert profile.diagnostics["unresolved_intervals"] == 0


def test_sampler_refines_on_backend_disagreement():
    def intensity_fn(x: float) -> FakeObserved:
        return FakeObserved(
            total=1.0,
            path_classes=["direct"],
            diagnostics={
                "termination_reason": "escape",
                "backend_disagreement": 1 if x > 0.5 else 0,
            },
        )

    profile = sample_radial_profile(
        0.0,
        1.0,
        intensity_fn,
        critical_points=[],
        options=SamplingOptions(initial_count=3, max_refine=1),
    )

    assert profile.diagnostics["refine_rounds"] == 1
    assert profile.diagnostics["sample_count"] > 3
    assert 0.25 not in profile.coordinates
    assert 0.75 in profile.coordinates
    assert profile.diagnostics["backend_disagreement_intervals"] == 1
    assert profile.diagnostics["shell_crossing_change_intervals"] == 0
    assert profile.diagnostics["unresolved_intervals"] == 0
