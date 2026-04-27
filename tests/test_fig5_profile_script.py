import csv
import subprocess
import sys


def _nearest(rows, b_value):
    return min(rows, key=lambda row: abs(float(row["b_over_M"]) - b_value))


def test_fig5_profile_script_outputs_bottom_row_profile(tmp_path):
    output = tmp_path / "fig5_bottom.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_fig5_profiles.py",
            "--profile",
            "bottom",
            "--output",
            str(output),
            "--samples",
            "160",
        ],
        check=True,
    )

    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert {
        "profile",
        "b_over_M",
        "Iobs_over_I0",
        "direct",
        "lensing_ring",
        "photon_ring",
        "intersection_count",
    } <= set(rows[0])
    assert {row["profile"] for row in rows} == {"bottom"}

    assert float(_nearest(rows, 2.0)["Iobs_over_I0"]) == 0.0
    assert float(_nearest(rows, 5.4)["Iobs_over_I0"]) > float(_nearest(rows, 4.0)["Iobs_over_I0"])
    assert float(_nearest(rows, 5.4)["lensing_ring"]) > 0.0
    assert float(_nearest(rows, 13.0)["Iobs_over_I0"]) < float(_nearest(rows, 6.5)["Iobs_over_I0"])


def test_fig5_profile_script_suppresses_expected_near_critical_warnings(tmp_path):
    output = tmp_path / "fig5_all.csv"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_fig5_profiles.py",
            "--profile",
            "all",
            "--output",
            str(output),
            "--samples",
            "300",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""


def test_fig5_profile_script_can_write_intensity_plot(tmp_path):
    output = tmp_path / "fig5_bottom.csv"
    plot = tmp_path / "fig5_bottom.png"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_fig5_profiles.py",
            "--profile",
            "bottom",
            "--output",
            str(output),
            "--plot-output",
            str(plot),
            "--samples",
            "160",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""
    assert output.exists()
    assert plot.exists()
    assert plot.stat().st_size > 0


def test_fig5_profile_script_can_write_axisymmetric_image(tmp_path):
    output = tmp_path / "fig5_all.csv"
    image = tmp_path / "fig5_images.png"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_fig5_profiles.py",
            "--profile",
            "all",
            "--output",
            str(output),
            "--image-output",
            str(image),
            "--samples",
            "160",
            "--image-size",
            "96",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""
    assert output.exists()
    assert image.exists()
    assert image.stat().st_size > 0
    assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
