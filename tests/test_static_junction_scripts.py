import json
import subprocess
import sys


def test_static_junction_profile_script_outputs_json_for_all_families(tmp_path):
    output = tmp_path / "profiles.json"
    image_dir = tmp_path / "images"

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_static_junction_profiles.py",
            "--output",
            str(output),
            "--samples",
            "4",
            "--max-refine",
            "0",
            "--max-order",
            "2",
            "--families",
            "schwarzschild",
            "rn",
            "rnds",
            "--compare-backends",
            "--image-output-dir",
            str(image_dir),
            "--image-size",
            "32",
            "--m-minus",
            "0.7",
            "--m-plus",
            "1.0",
            "--q-minus",
            "0.1",
            "--q-plus",
            "0.2",
            "--lambda-minus",
            "0.01",
            "--lambda-plus",
            "0.01",
            "--shell-radius",
            "5.0",
        ],
        check=True,
    )

    data = json.loads(output.read_text())
    assert {"schwarzschild", "rn", "rnds"} <= set(data["families"])
    assert len(data["families"]["schwarzschild"]["samples"]) == 4
    assert len(data["families"]["rn"]["samples"]) == 4
    assert len(data["families"]["rnds"]["samples"]) == 4
    assert {"rho", "alpha", "b_plus", "intensity", "intersections"} <= set(data["families"]["rn"]["samples"][0])
    for family in ["schwarzschild", "rn", "rnds"]:
        image = image_dir / f"{family}.png"
        assert image.exists()
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    rn_sample = data["families"]["rn"]["parameters"]
    assert rn_sample["m_minus"] == 0.7
    assert rn_sample["q_plus"] == 0.2
    assert "charge_mismatch_surface_source" in data["families"]["rn"]["warnings"]
    assert data["families"]["schwarzschild"]["samples"][0]["backend_comparison"]["comparison_backend"] == "hamiltonian"
    assert "backend_disagreement" in data["families"]["schwarzschild"]["samples"][0]["backend_comparison"]


def test_static_junction_validation_script_outputs_json(tmp_path):
    output = tmp_path / "validation.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_static_junction_validation.py",
            "--output",
            str(output),
            "--samples",
            "3",
        ],
        check=True,
    )

    data = json.loads(output.read_text())
    assert {"schwarzschild", "rn", "rnds"} <= set(data)
    assert len(data["schwarzschild"]["samples"]) == 3


def test_static_junction_profile_script_strict_physical_rejects_nonstatic_observer(tmp_path):
    output = tmp_path / "profiles.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_static_junction_profiles.py",
            "--output",
            str(output),
            "--families",
            "rnds",
            "--observer-radius",
            "50.0",
            "--strict-physical",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "strict physical" in result.stderr.lower()


def test_static_junction_profile_script_strict_physical_rejects_charge_mismatch(tmp_path):
    output = tmp_path / "profiles.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_static_junction_profiles.py",
            "--output",
            str(output),
            "--families",
            "rn",
            "--q-minus",
            "0.1",
            "--q-plus",
            "0.2",
            "--strict-physical",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "strict physical" in result.stderr.lower()
    assert "charge_mismatch_surface_source" in result.stderr


def test_static_junction_fig3_fig4_script_outputs_paper_reproduction_artifacts(tmp_path):
    output = tmp_path / "fig3_fig4.json"
    fig3 = tmp_path / "fig3.png"
    fig4_image = tmp_path / "fig4_image.png"
    fig4_trajectories = tmp_path / "fig4_trajectories.png"

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_static_junction_fig3_fig4.py",
            "--output",
            str(output),
            "--fig3-output",
            str(fig3),
            "--fig4-image-output",
            str(fig4_image),
            "--fig4-trajectories-output",
            str(fig4_trajectories),
            "--samples",
            "12",
            "--image-size",
            "32",
            "--trajectory-samples",
            "16",
            "--b-max",
            "6.0",
            "--max-order",
            "3",
        ],
        check=True,
    )

    data = json.loads(output.read_text())
    assert data["model"] == "schwarzschild_static_junction"
    assert data["parameters"]["m_minus"] == 0.1
    assert data["parameters"]["m_plus"] == 1.0
    assert data["parameters"]["shell_radius"] == 3.1
    assert data["source_model"]["name"] == "paper_eq_4_1"
    assert data["source_model"]["inner_horizon_radius"] == 0.2
    assert len(data["samples"]) >= 12
    assert {"b_plus", "intensity", "intersections", "shell_crossing_count"} <= set(data["samples"][0])
    assert any(sample["intersections"] for sample in data["samples"])
    assert any("g" in item and "r" in item for sample in data["samples"] for item in sample["intersections"])
    assert any(sample["shell_crossing_count"] > 0 for sample in data["samples"])
    for image in [fig3, fig4_image, fig4_trajectories]:
        assert image.exists()
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_rn_junction_image_script_outputs_parameter_sweep(tmp_path):
    output_dir = tmp_path / "rn_images"

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_rn_junction_images.py",
            "--output-dir",
            str(output_dir),
            "--samples",
            "3",
            "--max-refine",
            "0",
            "--max-order",
            "2",
            "--image-size",
            "24",
        ],
        check=True,
    )

    manifest = output_dir / "manifest.json"
    data = json.loads(manifest.read_text())
    assert data["model"] == "rn_static_junction_sweep"
    assert len(data["cases"]) >= 10
    assert len({case["name"] for case in data["cases"]}) == len(data["cases"])
    assert any("charge_mismatch_surface_source" in case["warnings"] for case in data["cases"])
    assert any(case["parameters"]["q_plus"] / case["parameters"]["m_plus"] > 0.9 for case in data["cases"])
    assert any(case["parameters"]["q_minus"] / case["parameters"]["m_minus"] > 0.9 for case in data["cases"])
    for case in data["cases"]:
        image = output_dir / case["image"]
        assert image.exists()
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert case["sample_count"] >= 3
        assert case["parameters"]["shell_radius"] > case["outer_photon_sphere_radius"]
