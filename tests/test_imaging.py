import numpy as np

from spherical_raytracing.imaging import (
    ImageGrid,
    RadialProfile,
    SamplingOptions,
    render_axisymmetric_image,
    sample_radial_profile,
)
from spherical_raytracing.sources import ObservedIntensity


def test_sampler_forces_critical_points_into_grid():
    profile = sample_radial_profile(
        coordinate_min=0.0,
        coordinate_max=10.0,
        intensity_fn=lambda x: x,
        critical_points=[5.0],
        options=SamplingOptions(initial_count=3, max_refine=0),
    )

    assert 5.0 in profile.coordinates
    assert profile.diagnostics["sample_count"] == len(profile.coordinates)


def test_sampler_refines_large_relative_jump():
    profile = sample_radial_profile(
        coordinate_min=0.0,
        coordinate_max=4.0,
        intensity_fn=lambda x: 0.0 if x < 2.0 else 10.0,
        critical_points=[],
        options=SamplingOptions(initial_count=3, max_refine=1, relative_jump_threshold=0.25),
    )

    assert len(profile.coordinates) > 3
    assert profile.diagnostics["refine_rounds"] == 1


def test_sampler_refines_when_path_class_changes_without_intensity_jump():
    def intensity_with_path_class(x):
        path_class = "direct" if x < 2.0 else "photon_ring"
        return ObservedIntensity(
            total=1.0,
            contributions=[1.0],
            radii=[10.0],
            redshift_weights=[1.0],
            regions=["external"],
            path_classes=[path_class],
            diagnostics={"termination_reason": "escape"},
        )

    profile = sample_radial_profile(
        coordinate_min=0.0,
        coordinate_max=4.0,
        intensity_fn=intensity_with_path_class,
        critical_points=[],
        options=SamplingOptions(
            initial_count=3,
            max_refine=1,
            absolute_gradient_threshold=100.0,
            relative_jump_threshold=100.0,
        ),
    )

    assert len(profile.coordinates) > 3
    assert profile.diagnostics["path_class_change_intervals"] > 0


def test_radial_profile_uses_monotone_interpolation():
    profile = RadialProfile(
        coordinates=np.array([0.0, 1.0, 2.0, 3.0]),
        intensities=np.array([4.0, 2.0, 1.0, 0.0]),
        diagnostics={"sample_count": 4},
    )

    assert profile.evaluate(np.array([0.0, 1.5, 3.0])).shape == (3,)
    assert profile.evaluate(np.array([5.0]))[0] == 0.0


def test_axisymmetric_render_rotates_radial_profile():
    profile = RadialProfile(
        coordinates=np.array([0.0, 1.0, 2.0]),
        intensities=np.array([1.0, 0.5, 0.0]),
        diagnostics={"sample_count": 3},
    )
    image = render_axisymmetric_image(profile, ImageGrid(width=5, height=5, radius=2.0))

    assert image.pixels.shape == (5, 5)
    assert image.pixels[2, 2] == 1.0
    assert image.pixels[0, 0] == 0.0
