import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_script_smoke_uses_manufactured_metrics_without_private_data(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_numerical_metric_perf.py",
            "--repeats",
            "1",
            "--no-u-at",
            "--real-data-dir",
            str(tmp_path / "missing-real-data"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )

    output = result.stdout
    assert "python=" in output
    assert "numpy=" in output
    assert "scipy=" in output
    assert "[manufactured_schwarzschild]" in output
    assert "[manufactured_rnds_bounded_outward]" in output
    assert "ray_count=" in output
    assert "category:captured=" in output
    assert "category:bounded_outward=" in output
    assert "median=" in output
    assert "p95=" in output
    assert "fallback=" in output
    assert "skipped:" in output
    assert "not found" in output


def test_benchmark_script_can_compare_fast_path_to_scalar_reference(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_numerical_metric_perf.py",
            "--repeats",
            "1",
            "--no-u-at",
            "--compare-reference",
            "--real-data-dir",
            str(tmp_path / "missing-real-data"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )

    output = result.stdout
    assert "[manufactured_schwarzschild:fast]" in output
    assert "[manufactured_schwarzschild:reference]" in output
    assert "solver:fast=" in output
    assert "solver:reference=" in output
    assert "fallback=" in output
