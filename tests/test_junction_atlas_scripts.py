import csv
import json
import importlib.util
from pathlib import Path
from PIL import Image
import subprocess
import sys


def _load_write_junction_atlas_report_module():
    module_path = Path("scripts/write_junction_atlas_report.py")
    spec = importlib.util.spec_from_file_location("write_junction_atlas_report", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_generate_junction_atlas_module():
    module_path = Path("scripts/generate_junction_atlas.py")
    spec = importlib.util.spec_from_file_location("generate_junction_atlas", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["generate_junction_atlas"] = module
    spec.loader.exec_module(module)
    return module


build_report = _load_write_junction_atlas_report_module().build_report
generate_junction_atlas = _load_generate_junction_atlas_module()


def test_junction_atlas_quick_preset_writes_manifest_and_phase_maps(tmp_path):
    output_dir = tmp_path / "atlas"

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_junction_atlas.py",
            "--preset",
            "quick",
            "--families",
            "rn",
            "rnds",
            "--grid-size",
            "4",
            "--samples",
            "4",
            "--image-size",
            "24",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["preset"] == "quick"
    assert set(manifest["families"]) == {"rn", "rnds"}
    assert manifest["physical_filter"]["horizon_buffer"] == 1e-3
    assert manifest["phase_map_summaries"]
    assert manifest["selected_cases"]
    assert manifest["selected_case_summaries"]
    assert "empty_representative_categories" in manifest
    assert manifest["skipped_representative_cases"] == []
    assert manifest["backend_comparison_summary"]["enabled"] is False
    assert "git_commit" in manifest
    assert {
        "fig_rn_mass_shell_phase.png",
        "fig_rnds_lambda_shell_phase.png",
    } <= set(manifest["figure_index"])

    for map_name in ["rn_mass_shell", "rnds_lambda_shell"]:
        csv_path = output_dir / "phase_maps" / f"{map_name}.csv"
        png_path = output_dir / "phase_maps" / f"{map_name}.png"
        stable_name = f"fig_{map_name}_phase.png"
        stable_path = output_dir / "figures" / stable_name
        figure_record = manifest["figure_index"][stable_name]
        assert csv_path.exists()
        assert png_path.exists()
        assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert stable_path.exists()
        assert stable_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert {"path", "case_id", "case_dir", "tag", "kind"} <= set(figure_record)
        assert figure_record["path"] == f"figures/{stable_name}"
        assert figure_record["kind"] == "phase_map"

        rows = list(csv.DictReader(csv_path.open(newline="")))
        assert rows
        assert {
            "family",
            "map_name",
            "m_minus",
            "m_plus",
            "q_minus",
            "q_plus",
            "lambda_minus",
            "lambda_plus",
            "shell_radius",
            "observer_radius",
            "admitted",
            "rejection_reasons",
            "warnings",
            "surface_energy_density",
            "surface_pressure",
            "q_jump",
            "lambda_jump",
            "photon_sphere_classification",
            "inner_photon_spheres",
            "outer_photon_spheres",
        } <= set(rows[0])


def test_junction_atlas_paper_preset_writes_report_scale_figures(tmp_path):
    output_dir = tmp_path / "atlas"

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_junction_atlas.py",
            "--preset",
            "paper",
            "--families",
            "rn",
            "--grid-size",
            "4",
            "--samples",
            "32",
            "--max-order",
            "2",
            "--image-size",
            "128",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["selected_case_summaries"]
    phase_image = Image.open(output_dir / "figures" / "fig_rn_mass_shell_phase.png")
    assert phase_image.size[0] >= 640
    assert phase_image.size[1] >= 640
    for summary in manifest["selected_case_summaries"]:
        case_dir = output_dir / summary["case_dir"]
        profile = Image.open(case_dir / "profile_paper.png")
        image = Image.open(case_dir / "image_paper.png")
        transfer = Image.open(case_dir / "transfer_redshift.png")
        assert profile.size[0] >= 900
        assert profile.size[1] >= 600
        assert image.size[0] >= 384
        assert image.size[1] >= 384
        assert transfer.size[0] >= 1200
        assert transfer.size[1] >= 500


def test_junction_atlas_rnds_focuses_mapped_inner_critical_curve(tmp_path):
    output_dir = tmp_path / "atlas"

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_junction_atlas.py",
            "--preset",
            "paper",
            "--families",
            "rnds",
            "--grid-size",
            "4",
            "--samples",
            "32",
            "--max-order",
            "4",
            "--image-size",
            "128",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    rnds_summaries = [
        summary for summary in manifest["selected_case_summaries"] if summary["family"] == "rnds"
    ]
    assert rnds_summaries
    found_high_order_case = False
    found_compact_high_order_branch = False
    for summary in rnds_summaries:
        profile = json.loads((output_dir / summary["case_dir"] / "profile_paper.json").read_text())
        order_counts: dict[int, int] = {}
        rhos_by_order: dict[int, list[float]] = {}
        for row in profile["samples"]:
            assert row["termination_reason"] != "near_critical"
            for item in row["intersections"]:
                order = int(item["m"])
                order_counts[order] = order_counts.get(order, 0) + 1
                rhos_by_order.setdefault(order, []).append(float(row["rho"]))
        if order_counts.get(2, 0) >= 20 and order_counts.get(3, 0) >= 5:
            found_high_order_case = True
        if order_counts.get(3, 0) >= 5:
            m3_rhos = rhos_by_order[3]
            if max(m3_rhos) - min(m3_rhos) < 0.1:
                found_compact_high_order_branch = True
        assert profile["sampling_diagnostics"]["dropped_near_critical_samples"] >= 0
        assert profile["sampling_diagnostics"]["mapped_inner_critical_screen_radii"]
        assert profile["sampling_diagnostics"]["focus_point_count"] > 0
    assert found_high_order_case
    assert found_compact_high_order_branch


def test_junction_atlas_quick_preset_writes_representative_cases(tmp_path):
    output_dir = tmp_path / "atlas"

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_junction_atlas.py",
            "--preset",
            "quick",
            "--families",
            "rn",
            "rnds",
            "--grid-size",
            "4",
            "--samples",
            "4",
            "--max-order",
            "2",
            "--image-size",
            "24",
            "--emissivity",
            "both",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["selected_cases"]
    assert manifest["figure_index"]

    for case_id in manifest["selected_cases"]:
        case_dir = output_dir / "cases" / case_id
        assert (case_dir / "diagnostics.json").exists()
        for name in [
            "profile_paper.csv",
            "profile_paper.json",
            "profile_paper.png",
            "profile_inverse_square.csv",
            "profile_inverse_square.json",
            "profile_inverse_square.png",
            "image_paper.png",
            "image_inverse_square.png",
            "transfer_redshift.png",
        ]:
            path = case_dir / name
            assert path.exists(), path
        for name in [
            "profile_paper.png",
            "profile_inverse_square.png",
            "image_paper.png",
            "image_inverse_square.png",
            "transfer_redshift.png",
        ]:
            assert (case_dir / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_junction_atlas_quick_preset_supports_inverse_square_only(tmp_path):
    output_dir = tmp_path / "atlas"

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_junction_atlas.py",
            "--preset",
            "quick",
            "--families",
            "rn",
            "--grid-size",
            "4",
            "--samples",
            "4",
            "--max-order",
            "2",
            "--image-size",
            "24",
            "--emissivity",
            "inverse_square",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["selected_case_summaries"]
    for summary in manifest["selected_case_summaries"]:
        assert summary["primary_emissivity"] == "inverse_square"
        assert summary["emissivity_modes"] == ["inverse_square"]
        case_dir = output_dir / "cases" / summary["case_id"]
        assert (case_dir / "profile_inverse_square.json").exists()
        assert not (case_dir / "profile_paper.json").exists()


def test_junction_atlas_rejects_too_few_profile_samples(tmp_path):
    output_dir = tmp_path / "atlas"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_junction_atlas.py",
            "--preset",
            "quick",
            "--families",
            "rn",
            "--samples",
            "1",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--samples must be at least 2" in result.stderr


def test_junction_atlas_rejects_explicit_zero_numeric_options(tmp_path):
    option_messages = [
        ("--samples", "--samples must be at least 2"),
        ("--max-order", "--max-order must be positive"),
        ("--image-size", "--image-size must be positive"),
    ]
    for option, message in option_messages:
        output_dir = tmp_path / option.removeprefix("--").replace("-", "_")
        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_junction_atlas.py",
                "--preset",
                "quick",
                "--families",
                "rn",
                option,
                "0",
                "--output-dir",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert message in result.stderr


def test_junction_atlas_inverse_square_compare_backends_reports_not_performed(tmp_path):
    output_dir = tmp_path / "atlas"

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_junction_atlas.py",
            "--preset",
            "quick",
            "--families",
            "rn",
            "--grid-size",
            "4",
            "--samples",
            "4",
            "--max-order",
            "2",
            "--image-size",
            "24",
            "--emissivity",
            "inverse_square",
            "--compare-backends",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["backend_comparison_summary"]["requested"] is True
    assert manifest["backend_comparison_summary"]["enabled"] is False
    assert all(
        summary["backend_comparison_summary"]["enabled"] is False
        for summary in manifest["selected_case_summaries"]
    )


def test_junction_atlas_backend_summary_counts_sampled_row_disagreements():
    rows = [
        {"backend_comparison": {"backend_disagreement": 0.0}},
        {"backend_comparison": {"backend_disagreement": 0.25}},
        {"backend_comparison": {"backend_disagreement": 0.0}},
        {"backend_comparison": {"backend_disagreement": 1.0}},
    ]

    summary = generate_junction_atlas._backend_comparison_summary(True, rows)

    assert summary == {
        "enabled": True,
        "backend_disagreement_intervals": 2,
    }


def test_junction_atlas_quick_preset_can_include_schwarzschild_reference(tmp_path):
    output_dir = tmp_path / "atlas_with_reference"

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_junction_atlas.py",
            "--preset",
            "quick",
            "--families",
            "rn",
            "--grid-size",
            "4",
            "--samples",
            "4",
            "--max-order",
            "2",
            "--image-size",
            "24",
            "--include-schwarzschild-reference",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["reference_artifacts"]
    assert "schwarzschild" in manifest["families"]
    for artifact_path in manifest["reference_artifacts"].values():
        assert (output_dir / artifact_path).exists(), artifact_path


def test_junction_atlas_schwarzschild_reference_preset_writes_fig3_to_fig8_artifacts(tmp_path):
    output_dir = tmp_path / "schwarzschild_reference"

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_junction_atlas.py",
            "--preset",
            "schwarzschild-reference",
            "--samples",
            "12",
            "--max-order",
            "3",
            "--image-size",
            "24",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["preset"] == "schwarzschild-reference"
    for artifact_path in manifest["reference_artifacts"].values():
        assert (output_dir / artifact_path).exists(), artifact_path

    expected = [
        "fig3_profile.json",
        "fig3_profile.png",
        "fig4_image.png",
        "fig4_trajectories.png",
        "fig5_sweep.json",
        "fig5_sweep.png",
        "fig6_images.png",
        "fig7_sweep.json",
        "fig7_sweep.png",
        "fig8_images.png",
    ]
    for name in expected:
        path = output_dir / "schwarzschild_reference" / name
        assert path.exists(), name
        if path.suffix == ".png":
            assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    fig5 = json.loads((output_dir / "schwarzschild_reference" / "fig5_sweep.json").read_text())
    fig7 = json.loads((output_dir / "schwarzschild_reference" / "fig7_sweep.json").read_text())
    assert [case["shell_radius"] for case in fig5["cases"]] == [5.0, 3.1, 2.9, 2.1]
    assert [case["shell_radius"] for case in fig7["cases"]] == [5.0, 3.1, 2.9, 2.6]
    assert fig7["m_minus"] == 0.8


def test_junction_atlas_schwarzschild_reference_preset_uses_defaults(tmp_path):
    output_dir = tmp_path / "schwarzschild_reference_defaults"

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_junction_atlas.py",
            "--preset",
            "schwarzschild-reference",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["preset"] == "schwarzschild-reference"
    for artifact_path in manifest["reference_artifacts"].values():
        assert (output_dir / artifact_path).exists(), artifact_path
    assert manifest["reference_artifacts"]["fig3_profile_json"] == "schwarzschild_reference/fig3_profile.json"
    assert (output_dir / "schwarzschild_reference" / "fig3_profile.json").exists()
    assert (output_dir / "schwarzschild_reference" / "fig4_image.png").exists()
    assert (output_dir / "schwarzschild_reference" / "fig7_sweep.json").exists()


def test_write_junction_atlas_report_creates_markdown_with_expected_sections(tmp_path):
    output_dir = tmp_path / "atlas"
    report = tmp_path / "docs" / "junction-atlas" / "rn-rnds-static-junction-atlas.md"

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_junction_atlas.py",
            "--preset",
            "quick",
            "--families",
            "rn",
            "rnds",
            "--grid-size",
            "4",
            "--samples",
            "4",
            "--max-order",
            "2",
            "--image-size",
            "24",
            "--emissivity",
            "both",
            "--include-schwarzschild-reference",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_junction_atlas_report.py",
            "--manifest",
            str(output_dir / "manifest.json"),
            "--output",
            str(report),
        ],
        check=True,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    text = report.read_text()
    for heading in [
        "# RN/RN-dS 静态 junction atlas",
        "## 摘要",
        "## 本次 atlas 研究什么",
        "## 几何设定与物理筛选",
        "## 参数 atlas 的生成方式",
        "## 报告图总览",
        "## phase map 分析",
        "## 代表性 case 分析",
        "## photon sphere 与 photon ring 的关系",
        "## 不同 emissivity 设定下的鲁棒性",
        "## Schwarzschild reference 复现分析",
        "## 局限与后续工作",
        "## 附录：所有产物索引",
        "### 图像索引",
        "### 产物索引",
    ]:
        assert heading in text
    assert all(f"`{case_id}`" in text for case_id in manifest["selected_cases"])
    assert "最终 selected case IDs 为：" in text
    assert "Selected case IDs:" not in text
    assert "## Overview" not in text
    assert sum("\u4e00" <= char <= "\u9fff" for char in text) > 300
    assert "共筛选" in text
    assert "admitted" in text
    assert "| case_id | family | category_tags | signal_score | peak_count | redshift_cusp_orders | v_shape_orders |" in text
    assert "| figure_name | report_path | kind | tag | source_case_dir |" in text
    assert "category_distances" in text
    assert "redshift_cusp_orders" in text
    assert "v_shape_orders" in text
    assert "主 profile 的实际采样点数" in text
    assert "mapped_inner_critical_screen_radii" in text
    assert "高阶分支" in text
    assert "低分辨率输出" not in text
    assert "快速 preset" not in text
    assert "quick preset" not in text
    assert "代表性 case 的 profile、image 与 transfer/redshift 图在后文逐个分析" in text
    assert "本次 selected cases 的峰数对比为：" in text
    assert "fig3_profile.png" in text
    assert "fig4_image.png" in text
    assert "fig4_trajectories.png" in text
    assert "fig5_sweep.png" in text
    assert "fig6_images.png" in text
    assert "fig7_sweep.png" in text
    assert "fig8_images.png" in text
    for figure_name, record in manifest["figure_index"].items():
        assert f"`{figure_name}`" in text
        if record["kind"] == "phase_map":
            assert f"![{figure_name}]" in text
    for summary in manifest["selected_case_summaries"]:
        assert f"**`{summary['case_id']}`** 是" in text
        for name in [
            "profile_paper.png",
            "image_paper.png",
            "transfer_redshift.png",
        ]:
            assert f"{summary['case_dir']}/{name}" in text
    for case_id in manifest["selected_cases"]:
        assert f"`{case_id}`" in text


def test_build_report_handles_partial_selected_case_summary_without_case_id():
    manifest = {
        "preset": "quick",
        "families": ["rn"],
        "phase_map_summaries": [
            {
                "row_count": 1,
                "admitted_count": 1,
                "rejected_count": 0,
            }
        ],
        "figure_index": {},
        "selected_cases": ["rn_partial_case"],
        "selected_case_summaries": [
            {
                "family": "rn",
                "category_tags": ["double"],
                "signal_score": 1.0,
                "profile_diagnostics": {
                    "peak_count": 2,
                    "redshift_cusp_orders": [],
                    "v_shape_orders": [],
                },
            }
        ],
    }

    text = build_report(manifest)

    assert "`rn_partial_case`" in text
    assert "最终 selected case IDs 为：" in text
    assert "Selected cases: ``" not in text


def test_build_report_includes_schwarzschild_reference_section_when_artifacts_exist():
    manifest = {
        "preset": "quick",
        "families": ["rn", "schwarzschild"],
        "phase_map_summaries": [],
        "figure_index": {},
        "selected_cases": [],
        "selected_case_summaries": [],
        "reference_artifacts": {
            "fig3_profile_json": "schwarzschild_reference/fig3_profile.json",
        },
    }

    text = build_report(manifest)

    assert "## Schwarzschild reference 复现分析" in text
    assert "### Schwarzschild reference 产物索引" in text
    assert "`fig3_profile_json`" in text
