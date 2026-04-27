import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

from spherical_raytracing.solvers import QuadTransferSolver


REFERENCE_RING_EDGES = [
    (0.072, 0.076),
    (1.077, 1.512),
    (2.974, 4.373),
    (4.414, 5.146),
    (4.995, 5.194),
]


def _load_lqg_fig3_script():
    script = Path("scripts/generate_lqg_fig3_profiles.py")
    spec = importlib.util.spec_from_file_location("generate_lqg_fig3_profiles", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_lqg_fig3_model_uses_core_through_trace_solver():
    module = _load_lqg_fig3_script()
    model = module.LQGFig3Model()

    assert isinstance(model.solver, QuadTransferSolver)


def test_lqg_fig3_script_writes_profile_image_and_caption_comparison(tmp_path):
    output = tmp_path / "lqg_fig3_profile.csv"
    plot = tmp_path / "lqg_fig3_profile.png"
    image = tmp_path / "lqg_fig3_image.png"
    comparison = tmp_path / "lqg_fig3_ring_edges.csv"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_lqg_fig3_profiles.py",
            "--output",
            str(output),
            "--plot-output",
            str(plot),
            "--image-output",
            str(image),
            "--comparison-output",
            str(comparison),
            "--samples",
            "48",
            "--image-size",
            "72",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""
    assert output.exists()
    assert plot.exists()
    assert image.exists()
    assert comparison.exists()
    assert plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    with comparison.open(newline="") as handle:
        comparison_rows = list(csv.DictReader(handle))

    assert len(comparison_rows) == 5
    for row, (expected_start, expected_end) in zip(comparison_rows, REFERENCE_RING_EDGES):
        assert abs(float(row["b_n_over_M"]) - expected_start) < 0.006
        assert abs(float(row["bprime_n_over_M"]) - expected_end) < 0.006

    with output.open(newline="") as handle:
        profile_rows = list(csv.DictReader(handle))

    assert profile_rows
    assert {
        "b_over_M",
        "Iobs_over_I0",
        "intersection_count",
        "r1_over_M",
        "r2_over_M",
        "r3_over_M",
        "phi_out",
        "phi_tot",
    } <= set(profile_rows[0])
    assert max(float(row["Iobs_over_I0"]) for row in profile_rows) > 0.0
