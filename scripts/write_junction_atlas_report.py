from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_MANIFEST = ROOT / "outputs" / "junction_atlas_schwarzschild_reference" / "manifest.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a markdown report from a junction atlas manifest.")
    parser.add_argument("--manifest", required=True, help="Path to the atlas manifest JSON.")
    parser.add_argument(
        "--output",
        default="docs/junction-atlas/rn-rnds-static-junction-atlas.md",
        help="Path to the markdown report to write.",
    )
    parser.add_argument(
        "--reference-manifest",
        help="Optional Schwarzschild reference manifest. Defaults to outputs/junction_atlas_schwarzschild_reference/manifest.json when present.",
    )
    return parser.parse_args()


def _load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _escape_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.replace("|", "\\|")


def _code_cell(value: object) -> str:
    return f"`{_escape_cell(value)}`"


def _format_case_tags(tags: Iterable[object]) -> str:
    items = [str(tag) for tag in tags]
    return ", ".join(items) if items else ""


def _format_float(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(number)
    return f"{number:.{digits}g}"


def _format_sequence(values: object) -> str:
    if values is None:
        return "无"
    if isinstance(values, str):
        return values if values else "无"
    if isinstance(values, Sequence):
        return ", ".join(str(value) for value in values) if values else "无"
    return str(values)


def _parameters(case: Mapping[str, object]) -> dict[str, object]:
    payload = case.get("parameters")
    return dict(payload) if isinstance(payload, Mapping) else {}


def _case_family(case: Mapping[str, object]) -> str:
    family = case.get("family")
    if family:
        return str(family)
    parameters = _parameters(case)
    if parameters.get("family"):
        return str(parameters["family"])
    return "unknown"


def _family_case_groups(cases: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = {"rn": [], "rnds": []}
    for case in cases:
        groups.setdefault(_case_family(case), []).append(case)
    return groups


def _markdown_relpath(path: Path, *, output_path: Path | None) -> str:
    if output_path is None:
        return path.as_posix()
    base = output_path.parent if output_path.suffix else output_path
    import os

    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return os.path.relpath(path.resolve(), base.resolve())


def _atlas_root(manifest: Mapping[str, object], *, manifest_path: Path | None) -> Path:
    if manifest_path is not None:
        return manifest_path.resolve().parent
    root = manifest.get("_manifest_root")
    if root:
        return Path(str(root)).resolve()
    return ROOT / "outputs" / "junction_atlas"


def _image_link(path: Path, alt: str, *, output_path: Path | None) -> str:
    return f"![{alt}]({_markdown_relpath(path, output_path=output_path)})"


def _load_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _profile_path(atlas_root: Path, case: Mapping[str, object], emissivity: str) -> Path:
    case_dir = case.get("case_dir") or f"cases/{case.get('case_id', '')}"
    return atlas_root / str(case_dir) / f"profile_{emissivity}.json"


def _case_artifact_path(atlas_root: Path, case: Mapping[str, object], name: str) -> Path:
    case_dir = case.get("case_dir") or f"cases/{case.get('case_id', '')}"
    return atlas_root / str(case_dir) / name


def _phase_summary_lines(manifest: Mapping[str, object]) -> list[str]:
    summaries = list(manifest.get("phase_map_summaries", []))
    row_count = sum(int(summary.get("row_count", 0)) for summary in summaries if isinstance(summary, Mapping))
    admitted_count = sum(int(summary.get("admitted_count", 0)) for summary in summaries if isinstance(summary, Mapping))
    rejected_count = sum(int(summary.get("rejected_count", 0)) for summary in summaries if isinstance(summary, Mapping))
    families = ", ".join(str(family) for family in manifest.get("families", []))
    lines = [
        f"本次 manifest 覆盖 `{families}`，共筛选 `{row_count}` 个参数点；其中 `{admitted_count}` 个通过物理筛选，`{rejected_count}` 个被拒绝。",
    ]
    if summaries:
        map_descriptions = []
        for summary in summaries:
            if not isinstance(summary, Mapping):
                continue
            counts = summary.get("photon_sphere_classifications", {})
            if isinstance(counts, Mapping):
                count_text = "、".join(f"`{key}`={value}" for key, value in sorted(counts.items()))
            else:
                count_text = "无分类统计"
            map_descriptions.append(
                f"`{summary.get('name', '')}`：`{summary.get('row_count', 0)}` 点，`{summary.get('admitted_count', 0)}` 个 admitted，分类为 {count_text}"
            )
        if map_descriptions:
            lines.append("phase map 的逐图统计为：" + "；".join(map_descriptions) + "。")
    return lines


def _phase_map_rows(manifest: Mapping[str, object], atlas_root: Path) -> dict[str, list[dict[str, str]]]:
    rows_by_name: dict[str, list[dict[str, str]]] = {}
    for summary in manifest.get("phase_map_summaries", []):
        if not isinstance(summary, Mapping):
            continue
        name = str(summary.get("name", ""))
        if not name or not summary.get("csv"):
            rows_by_name[name] = []
            continue
        csv_path = atlas_root / str(summary.get("csv", ""))
        rows_by_name[name] = _load_csv_rows(csv_path)
    return rows_by_name


def _count_warnings(rows: Sequence[Mapping[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        raw = row.get("warnings", "")
        if not raw:
            continue
        for warning in raw.split(";"):
            if warning:
                counts[warning] += 1
    return counts


def _phase_map_explanation(name: str, rows: Sequence[Mapping[str, str]]) -> str:
    if not rows:
        return f"`{name}` 是 manifest 记录的 phase map；当前报告没有读到对应 CSV 行，因此只保留 manifest 摘要而不追加数值区间解释。"
    if name == "rn_mass_shell":
        counts = Counter(row.get("photon_sphere_classification", "") for row in rows)
        shells = sorted({float(row["shell_radius"]) for row in rows if row.get("shell_radius")})
        masses = sorted({float(row["m_minus"]) for row in rows if row.get("m_minus")})
        return (
            f"`rn_mass_shell` 固定外侧 `m_plus=1` 与 `q_minus=q_plus=0.2`，扫描内侧质量 `m_minus` 和壳半径 `R_shell`。"
            f"这张图回答的是：壳把哪一侧的 RN `photon sphere` 留在可见静态区域中。当前网格的 `m_minus` 范围为 "
            f"`{_format_float(masses[0])}` 到 `{_format_float(masses[-1])}`，`R_shell` 范围为 `{_format_float(shells[0])}` 到 `{_format_float(shells[-1])}`；"
            f"分类计数为 `inner_only`={counts.get('inner_only', 0)}、`outer_only`={counts.get('outer_only', 0)}、`double`={counts.get('double', 0)}。"
            "在这组输出中，所有参数点都 admitted，因此图中颜色主要反映 photon-sphere 分区，而不是筛选边界。"
        )
    if name == "rnds_lambda_shell":
        counts = Counter(row.get("photon_sphere_classification", "") for row in rows)
        warnings = _count_warnings(rows)
        lambdas = sorted({float(row["lambda_plus"]) for row in rows if row.get("lambda_plus")})
        shells = sorted({float(row["shell_radius"]) for row in rows if row.get("shell_radius")})
        warning_text = "、".join(f"`{key}`={value}" for key, value in sorted(warnings.items())) or "无 warning"
        return (
            f"`rnds_lambda_shell` 固定 `m_minus=0.8`、`m_plus=1`、`q_minus=q_plus=0.2`、`lambda_minus=0.01`，扫描外侧 `lambda_plus` 与 `R_shell`。"
            f"它看的不是电荷相图，而是宇宙学常数跳变和 RN-dS clean static patch 下的可见 photon-sphere 分类。"
            f"当前 `lambda_plus` 范围为 `{_format_float(lambdas[0])}` 到 `{_format_float(lambdas[-1])}`，`R_shell` 范围为 `{_format_float(shells[0])}` 到 `{_format_float(shells[-1])}`；"
            f"全部 `{len(rows)}` 个点都被分类为 `inner_only`。warning 统计为 {warning_text}，说明这些 admitted 样本仍可携带真空能跳变或负表面能诊断。"
        )
    return f"`{name}` 是 manifest 记录的 phase map；CSV 中保存 admitted/rejected、warning、surface source 与 photon-sphere 分类。"


def _figure_index_rows(figure_index: Mapping[str, Mapping[str, object]]) -> list[str]:
    rows = []
    for figure_name in sorted(figure_index):
        record = figure_index[figure_name]
        rows.append(
            "| "
            + " | ".join(
                [
                    _code_cell(figure_name),
                    _code_cell(record.get("path", "")),
                    _code_cell(record.get("kind", "")),
                    _code_cell(record.get("tag", "")),
                    _code_cell(record.get("case_dir", "")),
                ]
            )
            + " |"
        )
    return rows


def _figure_record_caption(figure_name: str, record: Mapping[str, object]) -> str:
    kind = str(record.get("kind", ""))
    tag = str(record.get("tag", ""))
    case_id = str(record.get("case_id", ""))
    if kind == "phase_map":
        if tag == "rn_mass_shell":
            return (
                f"`{figure_name}` 是 `rn_mass_shell` phase map 的报告稳定副本。它把横轴 `m_minus`、纵轴 `R_shell` 的 RN 网格按 "
                "`photon_sphere_classification` 着色，用来定位 inner-only、outer-only 和 double 几何区。"
            )
        if tag == "rnds_lambda_shell":
            return (
                f"`{figure_name}` 是 `rnds_lambda_shell` phase map 的报告稳定副本。它把横轴 `lambda_plus`、纵轴 `R_shell` 的 RN-dS 网格按 "
                "`photon_sphere_classification` 着色，同时应结合 CSV 中的 `lambda_jump` 与 warning 读解。"
            )
        return f"`{figure_name}` 是 `{tag}` phase map 的报告稳定副本。"
    if kind == "representative_case":
        return (
            f"`{figure_name}` 是 `{case_id}` 在主 emissivity 下的代表图像副本，对应标签 `{tag}`。"
            "它用于快速比较不同代表类别的环状亮度结构；定量峰数仍以同一 case 目录下的 profile JSON/CSV 为准。"
        )
    return f"`{figure_name}` 是 manifest `figure_index` 记录的报告图。"


def _report_figure_gallery(
    figure_index: Mapping[str, Mapping[str, object]],
    atlas_root: Path,
    *,
    output_path: Path | None,
) -> list[str]:
    if not figure_index:
        return []
    lines = ["## 报告图总览", ""]
    lines.append(
        "`outputs/junction_atlas/figures/` 存放报告用稳定文件名。正文在这里只预览 phase map；代表性 case 的 profile、image 与 transfer/redshift 图在后文逐个分析。`fig_case_*` 是同一 case 主 emissivity 图像按标签复制出的索引图，避免正文重复嵌入，完整列表放在附录图像索引。"
    )
    lines.append("")
    for figure_name in sorted(figure_index):
        record = figure_index[figure_name]
        if not isinstance(record, Mapping):
            continue
        if record.get("kind") != "phase_map":
            continue
        path = atlas_root / str(record.get("path", ""))
        if not path.exists() or path.suffix.lower() != ".png":
            continue
        lines.append(_image_link(path, figure_name, output_path=output_path))
        lines.append("")
        lines.append(_figure_record_caption(figure_name, record))
        lines.append("")
    return lines


def _selected_case_rows(cases: list[dict[str, object]]) -> list[str]:
    rows = []
    for case in cases:
        diagnostics = dict(case.get("profile_diagnostics", {}))
        rows.append(
            "| "
            + " | ".join(
                [
                    _code_cell(case.get("case_id", "")),
                    _code_cell(_case_family(case)),
                    _escape_cell(_format_case_tags(case.get("category_tags", []))),
                    _escape_cell(_format_float(case.get("signal_score", ""))),
                    _escape_cell(diagnostics.get("peak_count", "")),
                    _escape_cell(_format_sequence(diagnostics.get("redshift_cusp_orders", []))),
                    _escape_cell(_format_sequence(diagnostics.get("v_shape_orders", []))),
                ]
            )
            + " |"
        )
    return rows


def _reference_artifact_rows(reference_artifacts: Mapping[str, object]) -> list[str]:
    rows = []
    for artifact_name in sorted(reference_artifacts):
        rows.append(
            "| "
            + " | ".join(
                [
                    _code_cell(artifact_name),
                    _code_cell(reference_artifacts[artifact_name]),
                ]
            )
            + " |"
        )
    return rows


def _peak_summary(diagnostics: Mapping[str, object]) -> str:
    peaks = diagnostics.get("peaks", [])
    if not isinstance(peaks, Sequence) or not peaks:
        return "没有检测到局部强度峰。"
    fragments = []
    for peak in peaks:
        if not isinstance(peak, Mapping):
            continue
        fragments.append(
            f"`rho={_format_float(peak.get('x'))}`、`I={_format_float(peak.get('y'))}`、prominence `{_format_float(peak.get('prominence'), 3)}`"
        )
    return "检测到的峰为：" + "；".join(fragments) + "。"


def _order_stats(samples: object) -> dict[int, dict[str, float | int]]:
    stats: dict[int, dict[str, float | int]] = {}
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        return stats
    by_order: dict[int, list[float]] = {}
    for row in samples:
        if not isinstance(row, Mapping):
            continue
        rho = float(row.get("rho", 0.0))
        intersections = row.get("intersections", [])
        if not isinstance(intersections, Sequence) or isinstance(intersections, (str, bytes)):
            continue
        for item in intersections:
            if not isinstance(item, Mapping):
                continue
            order = int(item.get("m", 0))
            if order <= 0:
                continue
            by_order.setdefault(order, []).append(rho)
    for order, values in by_order.items():
        sorted_values = sorted(values)
        gaps = [right - left for left, right in zip(sorted_values[:-1], sorted_values[1:])]
        stats[order] = {
            "count": len(sorted_values),
            "rho_min": sorted_values[0],
            "rho_max": sorted_values[-1],
            "max_gap": max(gaps) if gaps else 0.0,
        }
    return stats


def _order_stats_text(stats: Mapping[int, Mapping[str, float | int]]) -> str:
    fragments = []
    for order, payload in sorted(stats.items()):
        fragments.append(
            f"`m={order}`:{payload.get('count', 0)} 点，`rho=[{_format_float(payload.get('rho_min'))}, {_format_float(payload.get('rho_max'))}]`"
        )
    return "；".join(fragments) or "无 disk crossing"


def _high_order_window_text(stats: Mapping[int, Mapping[str, float | int]]) -> str:
    high = [payload for order, payload in stats.items() if order >= 2]
    if not high:
        return ""
    rho_min = min(float(payload["rho_min"]) for payload in high)
    rho_max = max(float(payload["rho_max"]) for payload in high)
    total = sum(int(payload["count"]) for payload in high)
    return f"`m>=2` 共 `{total}` 个交点样本，集中在 `rho=[{_format_float(rho_min)}, {_format_float(rho_max)}]`"


def _category_glossary_lines() -> list[str]:
    return [
        "`category_tags` 是代表性 case 的标签集合。`inner_only`、`outer_only` 和 `double` 来自 `classify_photon_spheres`：只统计壳内侧且在壳内的 photon sphere，以及壳外侧且在壳外的 photon sphere；`double` 表示两侧都保留。`single_sphere_double_peak` 是 profile 后验标签，表示几何上只有一侧 photon sphere，但强度曲线至少有两个峰；这正是参考文献强调的 photon sphere 与 photon ring 非一一对应现象。",
        "`category_distances` 是参数点到各代表类别锚点的归一化距离，只用于确定性选样；它不是几何不变量。`signal_score` 也是选择代表性样本用的诊断分数，不是观测显著性的物理量。脚本把 photon-sphere 分类、warning 数量、`Q_jump` 与 `Lambda_jump` 加权合成，用来在同类候选中排序。",
        "`peak_count` 来自 `diagnose_profile` 对主 emissivity profile 的局部极大值计数；`redshift_cusp_orders` 记录哪几个 disk-crossing order 的发射半径正好触及壳半径，因而可能产生红移 cusp；`v_shape_orders` 记录哪几个 order 的 transfer function `r(rho)` 出现足够显著的局部极值，可作为 V 形结构的数值指示。空列表表示当前采样没有检测到，不等于严格证明物理上不存在。",
    ]


def _case_parameter_text(parameters: Mapping[str, object]) -> str:
    ordered = [
        "m_minus",
        "m_plus",
        "q_minus",
        "q_plus",
        "lambda_minus",
        "lambda_plus",
        "shell_radius",
        "observer_radius",
    ]
    return "，".join(f"`{key}={_format_float(parameters.get(key))}`" for key in ordered if key in parameters)


def _case_explanation(case: Mapping[str, object], atlas_root: Path) -> list[str]:
    diagnostics = dict(case.get("profile_diagnostics", {}))
    parameters = _parameters(case)
    tags = [str(tag) for tag in case.get("category_tags", [])]
    case_id = str(case.get("case_id", ""))
    modes = [str(mode) for mode in case.get("emissivity_modes", [])]
    primary = str(case.get("primary_emissivity") or (modes[0] if modes else "paper"))
    shell_crossing = diagnostics.get("shell_crossing_count_max", 0)
    lines = [
        f"**`{case_id}`** 是 `{_case_family(case)}` 代表样本，参数为 {_case_parameter_text(parameters)}。它的 `category_tags` 为 `{_format_case_tags(tags)}`，`signal_score={_format_float(case.get('signal_score'))}`；主 profile 使用 `{primary}` emissivity，最大壳穿越次数为 `{shell_crossing}`。",
        f"{_peak_summary(diagnostics)}`redshift_cusp_orders={_format_sequence(diagnostics.get('redshift_cusp_orders', []))}`，`v_shape_orders={_format_sequence(diagnostics.get('v_shape_orders', []))}`。",
    ]
    if "single_sphere_double_peak" in tags:
        lines.append(
            "这个 case 被标记为 `single_sphere_double_peak`，含义是：按 photon-sphere 几何分类只剩一侧 photon sphere，但强度 profile 出现了两个或更多局部峰。这里可以说存在“ring/peak 结构多于 photon-sphere 计数”的数值证据；峰的成因仍需结合 transfer/redshift 图与更密采样判断。"
        )
    if "double" in tags and int(diagnostics.get("peak_count", 0) or 0) <= 2:
        lines.append(
            "这个 case 的几何标签为 `double`，表示壳内和壳外都存在可见 photon sphere。当前 profile 检测到的峰数没有简单等于“两个 photon sphere 加两个峰”的教科书图像，提醒我们 ring 可见性还受壳折射、红移权重、发射模型和采样分辨率控制。"
        )
    if "large_lambda_jump" in tags:
        lines.append(
            "`large_lambda_jump` 表示壳两侧真空能密度不同，`Lambda_plus-Lambda_minus` 在代表性选择中给出较高诊断权重。该标签不是 rejection，而是提示壳面源包含由 cosmological constant 跳变带来的物理解释负担。"
        )

    paper_profile = _load_json_if_exists(_profile_path(atlas_root, case, "paper"))
    inverse_profile = _load_json_if_exists(_profile_path(atlas_root, case, "inverse_square"))
    if paper_profile:
        sampling = dict(paper_profile.get("sampling_diagnostics", {}))
        samples = paper_profile.get("samples", [])
        order_stats = _order_stats(samples)
        order_text = _order_stats_text(order_stats)
        sample_count = len(samples) if isinstance(samples, Sequence) and not isinstance(samples, (str, bytes)) else "NA"
        lines.append(
            f"主 profile 的实际采样点数为 `{sample_count}`，其中 disk-crossing order 覆盖为 {order_text}；`dropped_near_critical_samples={sampling.get('dropped_near_critical_samples', 0)}`。临界点本身被剔除，以免把 `near_critical` 终止点误画成强度归零。"
        )
        if _case_family(case) == "rnds":
            mapped = sampling.get("mapped_inner_critical_screen_radii", [])
            mapped_text = _format_sequence([_format_float(value) for value in mapped] if isinstance(mapped, Sequence) else mapped)
            high_order_text = _high_order_window_text(order_stats)
            if high_order_text:
                lines.append(
                    f"这个 RN-dS case 的高阶分支满足：{high_order_text}。这些点围绕内侧 photon sphere 经壳折射映射到外侧屏幕后的临界半径；当前 profile JSON 记录的 `mapped_inner_critical_screen_radii={mapped_text}`。因此 `transfer_redshift` 图中接近竖直的 m=2/3/4 分支和 inset 是临界曲线附近的强弯曲结构，不应解读为 m=3 没有计算或采样断裂。"
                )
            else:
                lines.append(
                    f"这个 RN-dS case 的 profile JSON 记录 `mapped_inner_critical_screen_radii={mapped_text}`；高阶分支是否可见仍需看 `transfer_redshift` 图和原始 JSON，而不能只凭主图线宽判断。"
                )
    if paper_profile and inverse_profile:
        paper_peaks = int(dict(paper_profile.get("profile_diagnostics", {})).get("peak_count", 0))
        inverse_peaks = int(dict(inverse_profile.get("profile_diagnostics", {})).get("peak_count", 0))
        lines.append(
            f"emissivity 对比中，`paper` profile 检测到 `{paper_peaks}` 个峰，`inverse_square` profile 检测到 `{inverse_peaks}` 个峰。若二者峰数一致，说明 peak 数量对本次发射模型替换较稳健；若峰位或峰高不同，则应把差异归因于发射权重，而不是几何 photon-sphere 分类改变。"
        )
    return lines


def _case_figure_block(case: Mapping[str, object], atlas_root: Path, *, output_path: Path | None) -> list[str]:
    lines = []
    artifacts = [
        ("profile_paper.png", "paper emissivity 下的径向强度 profile，横轴是 observer screen radius `rho`，纵轴是观测强度。它直接支撑 `peak_count` 与峰位置。"),
        ("image_paper.png", "由 `profile_paper` 旋转渲染得到的轴对称图像；亮环对应 profile 中的局部增强，但是否是 photon-sphere ring 需要结合几何分类判断。"),
        ("transfer_redshift.png", "`transfer_redshift` 左栏给出不同 disk-crossing order 的发射半径 `r(rho)`，右栏给出红移因子 `g(rho)`；它用来检查壳附近 cusp 与 V 形 transfer 结构。"),
    ]
    if "inverse_square" in case.get("emissivity_modes", []):
        artifacts.extend(
            [
                ("profile_inverse_square.png", "`inverse_square` emissivity 下的 profile，用作发射模型鲁棒性检查。"),
                ("image_inverse_square.png", "`inverse_square` emissivity 的渲染图像；与 paper 图相比，几何位置通常比亮度归一化更有解释价值。"),
            ]
        )
    for filename, caption in artifacts:
        path = _case_artifact_path(atlas_root, case, filename)
        if path.exists():
            lines.append(_image_link(path, f"{case.get('case_id', '')} {filename}", output_path=output_path))
            lines.append("")
            lines.append(caption)
            lines.append("")
    return lines


def _reference_manifest_from_args(reference_manifest: Path | None) -> tuple[dict[str, object], Path | None]:
    path = reference_manifest
    if path is None and DEFAULT_REFERENCE_MANIFEST.exists():
        path = DEFAULT_REFERENCE_MANIFEST
    if path is None or not path.exists():
        return {}, None
    manifest = _load_manifest(path)
    return manifest, path.resolve().parent


def _merge_reference_artifacts(
    manifest: Mapping[str, object],
    reference_manifest: Mapping[str, object],
) -> dict[str, object]:
    artifacts: dict[str, object] = {}
    if isinstance(manifest.get("reference_artifacts"), Mapping):
        artifacts.update(dict(manifest["reference_artifacts"]))
    if isinstance(reference_manifest.get("reference_artifacts"), Mapping):
        artifacts.update(dict(reference_manifest["reference_artifacts"]))
    return artifacts


def _reference_root(manifest: Mapping[str, object], manifest_root: Path | None, reference_manifest_root: Path | None) -> Path:
    if manifest.get("reference_artifacts"):
        return manifest_root or ROOT / "outputs" / "junction_atlas"
    return reference_manifest_root or DEFAULT_REFERENCE_MANIFEST.parent


def _reference_json_summary(path: Path) -> dict[str, object]:
    payload = _load_json_if_exists(path)
    if not payload:
        return {}
    return payload


def _reference_case_stats(payload: Mapping[str, object]) -> str:
    cases = payload.get("cases", [])
    if not isinstance(cases, Sequence):
        return ""
    pieces = []
    for case in cases:
        if not isinstance(case, Mapping):
            continue
        diagnostics = dict(case.get("profile_diagnostics", {}))
        pieces.append(
            f"`R_shell={_format_float(case.get('shell_radius'))}`: `peak_count={diagnostics.get('peak_count', 'NA')}`，`v_shape_orders={_format_sequence(diagnostics.get('v_shape_orders', []))}`"
        )
    return "；".join(pieces)


def _schwarzschild_reference_section(
    manifest: Mapping[str, object],
    reference_manifest: Mapping[str, object],
    *,
    manifest_root: Path | None,
    reference_manifest_root: Path | None,
    output_path: Path | None,
) -> list[str]:
    artifacts = _merge_reference_artifacts(manifest, reference_manifest)
    if not artifacts:
        return []
    root = _reference_root(manifest, manifest_root, reference_manifest_root)
    lines = ["## Schwarzschild reference 复现分析", ""]
    lines.append(
        "参考论文的静态 Schwarzschild-Schwarzschild junction 给本 atlas 提供了校准基线：静态壳会带来红移 cusp、壳折射导致的 V 形 transfer function，以及 photon sphere 与 photon ring/peak 非一一对应。这里的复现使用 `m_plus=1`、观测者 `r_obs=50`，并沿用 paper-style thin-disk emissivity。"
    )
    lines.append("")

    ordered = [
        ("fig3_profile_png", "Fig. 3 profile：同时画出 `I_obs(b)`、不同 disk-crossing order 的 transfer function `r(b)` 与红移因子 `g_n(b)`。它是判断红移 cusp、V 形 transfer 与多峰结构的核心图。"),
        ("fig4_trajectories_png", "Fig. 4 trajectories：对比穿壳与不穿壳的光线轨道，直接显示壳处折射如何改变第二次盘交点半径。"),
        ("fig4_image_png", "Fig. 4 image：把 Fig. 3 的 profile 渲染成图像；参考论文指出其中明亮内环来自内侧 photon sphere，较暗外环可在没有外侧 photon sphere 的情况下出现。"),
        ("fig5_sweep.png", "Fig. 5 sweep：固定 `m_minus=0.1`，扫描 `R_shell=5, 3.1, 2.9, 2.1` 的 profile/transfer。它展示壳向内移动时，从内侧 ring 主导到双峰增强的连续变化。"),
        ("fig6_images.png", "Fig. 6 images：对应 Fig. 5 的图像序列；重点是内环外移、变暗，以及外环逐渐显现。"),
        ("fig7_sweep.png", "Fig. 7 sweep：固定 `m_minus=0.8`，扫描 `R_shell=5, 3.1, 2.9, 2.6`。参考论文强调最后一个点仍可有 double photon sphere，但观测 profile 只剩单峰。"),
        ("fig8_images.png", "Fig. 8 images：对应 Fig. 7 的图像序列；它显示内外环亮度接近后，内环在 `R_shell=2.6` 处消失。"),
    ]
    for key, caption in ordered:
        rel = artifacts.get(key)
        if rel is None and key.endswith("_png"):
            rel = artifacts.get(key.replace("_png", ".png"))
        if rel:
            lines.append(_image_link(root / str(rel), key, output_path=output_path))
            lines.append("")
            lines.append(caption)
            lines.append("")

    fig3_key = artifacts.get("fig3_profile_json")
    if fig3_key:
        fig3 = _reference_json_summary(root / str(fig3_key))
        diagnostics = dict(fig3.get("profile_diagnostics", {}))
        params = dict(fig3.get("parameters", {}))
        lines.append(
            f"`fig3_profile.json` 记录复现样本 {_case_parameter_text(params)}；本次输出检测到 `peak_count={diagnostics.get('peak_count', 'NA')}`，`v_shape_orders={_format_sequence(diagnostics.get('v_shape_orders', []))}`，`redshift_cusp_orders={_format_sequence(diagnostics.get('redshift_cusp_orders', []))}`。这些数值是脚本诊断，不替代肉眼读图；特别是 cusp 检测要求采样点精确落在壳半径，可能偏保守。"
        )
        lines.append("")
    for sweep_name in ["fig5_sweep.json", "fig7_sweep.json"]:
        rel = artifacts.get(sweep_name)
        if not rel:
            continue
        payload = _reference_json_summary(root / str(rel))
        stats = _reference_case_stats(payload)
        if stats:
            lines.append(f"`{sweep_name}` 的逐半径诊断为：{stats}。")
            lines.append("")

    lines.append("这些 reference 结果支撑本文的解释边界：我们可以可靠地讨论“壳折射和红移改变 ring/peak 可见性”，但不能只从 `peak_count` 反推出 photon sphere 数量。")
    lines.append("")
    lines.append("### Schwarzschild reference 产物索引")
    lines.append("")
    lines.append("| artifact_name | report_path |")
    lines.append("| --- | --- |")
    lines.extend(_reference_artifact_rows(artifacts))
    lines.append("")
    return lines


def build_report(
    manifest: dict[str, object],
    *,
    manifest_path: Path | None = None,
    output_path: Path | None = None,
    reference_manifest_path: Path | None = None,
) -> str:
    atlas_root = _atlas_root(manifest, manifest_path=manifest_path)
    reference_manifest, reference_manifest_root = _reference_manifest_from_args(reference_manifest_path)
    selected_case_ids = [str(case_id) for case_id in manifest.get("selected_cases", [])]
    selected_case_summaries = [dict(case) for case in manifest.get("selected_case_summaries", []) if isinstance(case, Mapping)]
    families = ", ".join(str(family) for family in manifest.get("families", []))
    phase_rows = _phase_map_rows(manifest, atlas_root)
    figure_index = dict(manifest.get("figure_index", {})) if isinstance(manifest.get("figure_index"), Mapping) else {}

    lines: list[str] = []
    lines.append("# RN/RN-dS 静态 junction atlas")
    lines.append("")
    lines.append("## 摘要")
    lines.append("")
    lines.append(
        f"本文档把 `outputs/junction_atlas/manifest.json` 记录的 `{families}` 静态薄壳 junction 扫描整理成一份中文研究札记。目标不是给出最终高分辨率论文图，而是说明当前 atlas 如何筛选物理参数、如何分类 `photon sphere`，以及这些几何分类怎样在 `photon ring` 或强度峰结构中被保留、削弱或改写。"
    )
    if manifest.get("preset"):
        lines.append(f"本次 atlas preset 为 `{manifest['preset']}`；`grid_size={manifest.get('grid_size', 'NA')}`、`samples={manifest.get('samples', 'NA')}`、`max_order={manifest.get('max_order', 'NA')}`、`image_size={manifest.get('image_size', 'NA')}`。")
    if manifest.get("git_commit"):
        lines.append(f"生成 manifest 的 git commit 为 `{manifest['git_commit']}`。")
    lines.append("")

    lines.append("## 本次 atlas 研究什么")
    lines.append("")
    lines.append(
        "本 atlas 研究的是同族静态球对称 junction：RN-RN 与 RN-dS-RN-dS。壳位于固定半径 `R_shell`，内外 metric 可以有不同质量、电荷和宇宙学常数。光线在每一侧按该侧 metric 传播，穿过壳时因为静态观察者基底和 Killing 能量的匹配发生折射与能量重标定；随后用薄、光学薄的赤道盘发射模型计算观测强度。"
    )
    lines.append(
        "参考论文研究 Schwarzschild-Schwarzschild junction，并指出静态壳的三个信号：红移在壳处出现 cusp、transfer function `r(b)` 可出现 V 形、以及 `photon sphere` 与最终 `photon ring`/峰结构失去一一对应。本 atlas 把这套问题推广到 RN 和 RN-dS；当前输出是有限网格的数值 atlas，因此结论应读作研究札记，而不是参数空间穷尽。"
    )
    lines.append("")

    lines.append("## 几何设定与物理筛选")
    lines.append("")
    filter_payload = manifest.get("physical_filter", {})
    if isinstance(filter_payload, Mapping):
        lines.append(
            f"物理筛选使用 `horizon_buffer={filter_payload.get('horizon_buffer', '')}`，并设置 `require_clean_rnds_patch={filter_payload.get('require_clean_rnds_patch', '')}`。筛选要求壳在两侧都是静态的，外侧观察者在壳外且静态，RN/RN-dS 不接受裸奇点样本，RN-dS 还要求存在可识别的 black-hole-to-cosmological clean static patch。"
        )
    else:
        lines.append("manifest 记录了物理筛选配置；报告从该配置与 phase-map CSV 中读取 admitted/rejected 和 warning。")
    lines.append(
        "`surface_energy_density`、`surface_pressure`、`q_jump=Q_plus-Q_minus` 与 `lambda_jump=Lambda_plus-Lambda_minus` 被写入 CSV/JSON。负表面能、电荷跳变和真空能跳变默认是 warning 或解释负担，不自动等同于参数点无效；非静态壳、观察者不可静态、裸奇点和 RN-dS clean patch 失败才进入 rejection。"
    )
    lines.append("")

    lines.append("## 参数 atlas 的生成方式")
    lines.append("")
    lines.extend(_phase_summary_lines(manifest))
    lines.append(
        "生成流程分两层：先用 phase map 做低成本几何扫描，记录每个网格点的 admitted 状态、warning、壳面源和 `photon_sphere_classification`；再从 admitted 候选中按 `category_tags`、到锚点的距离和 `signal_score` 选择代表性 case，运行 profile、image 与 transfer/redshift 图。"
    )
    if selected_case_ids:
        lines.append("最终 selected case IDs 为：" + "，".join(_code_cell(case_id) for case_id in selected_case_ids) + "。")
    if manifest.get("empty_representative_categories"):
        lines.append("当前输出没有覆盖的代表类别为：" + _format_sequence(manifest.get("empty_representative_categories")) + "。这通常意味着当前网格与代表性选择没有采到该结构，不能解释为该结构在连续参数空间中不存在。")
    lines.append("")

    lines.extend(_report_figure_gallery(figure_index, atlas_root, output_path=output_path))

    lines.append("## phase map 分析")
    lines.append("")
    for summary in manifest.get("phase_map_summaries", []):
        if not isinstance(summary, Mapping):
            continue
        name = str(summary.get("name", ""))
        png = atlas_root / str(summary.get("png", ""))
        stable = atlas_root / "figures" / f"fig_{name}_phase.png"
        figure_path = stable if stable.exists() else png
        if figure_path.exists():
            lines.append(_image_link(figure_path, f"{name} phase map", output_path=output_path))
            lines.append("")
        lines.append(_phase_map_explanation(name, phase_rows.get(name, [])))
        lines.append("")

    lines.append("## 代表性 case 分析")
    lines.append("")
    lines.extend(_category_glossary_lines())
    lines.append("")
    lines.append("| case_id | family | category_tags | signal_score | peak_count | redshift_cusp_orders | v_shape_orders |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    lines.extend(_selected_case_rows(selected_case_summaries))
    lines.append("")
    for family, cases in _family_case_groups(selected_case_summaries).items():
        if not cases:
            continue
        title = "RN case 组" if family == "rn" else "RN-dS case 组" if family == "rnds" else f"{family} case 组"
        lines.append(f"### {title}")
        lines.append("")
        for case in cases:
            lines.extend(_case_explanation(case, atlas_root))
            lines.append("")
            lines.extend(_case_figure_block(case, atlas_root, output_path=output_path))

    lines.extend(
        _schwarzschild_reference_section(
            manifest,
            reference_manifest,
            manifest_root=atlas_root,
            reference_manifest_root=reference_manifest_root,
            output_path=output_path,
        )
    )

    lines.append("## photon sphere 与 photon ring 的关系")
    lines.append("")
    lines.append(
        "在脚本中，`photon_sphere_classification` 是纯几何分类：把内侧 metric 的 photon sphere 中满足 `r<R_shell` 的部分记为 inner，把外侧 metric 中满足 `r>R_shell` 的部分记为 outer。它告诉我们几何上哪些不稳定圆光轨没有被壳切掉。"
    )
    lines.append(
        "但最终图像中的 ring 或 profile peak 还依赖三个额外因素：壳折射如何把内侧临界 impact parameter 映射到外侧屏幕半径，红移因子如何压低或增强某一阶盘交点，以及 emissivity 如何给不同发射半径加权。因此 `inner_only` 可以出现双峰，`double` 也可能不呈现两个清晰可分的亮环。当前 selected cases 中的 `single_sphere_double_peak` 标签正是这种非一一对应的数值证据。"
    )
    lines.append("")

    lines.append("## 不同 emissivity 设定下的鲁棒性")
    lines.append("")
    lines.append(
        "`paper` emissivity 使用参考论文风格的薄盘模型：从 family-aware inner edge 起按近似 `(r-r_in+1)^-3` 衰减；`inverse_square` 则使用 `I_em ~ 1/r^2`。前者更贴近 Schwarzschild reference，后者用于检查结论是否只是某个发射权重造成。"
    )
    comparison_fragments = []
    for case in selected_case_summaries:
        paper_profile = _load_json_if_exists(_profile_path(atlas_root, case, "paper"))
        inverse_profile = _load_json_if_exists(_profile_path(atlas_root, case, "inverse_square"))
        if not paper_profile or not inverse_profile:
            continue
        paper_diag = dict(paper_profile.get("profile_diagnostics", {}))
        inverse_diag = dict(inverse_profile.get("profile_diagnostics", {}))
        comparison_fragments.append(
            f"`{case.get('case_id', '')}`: paper `{paper_diag.get('peak_count', 'NA')}` 峰，inverse_square `{inverse_diag.get('peak_count', 'NA')}` 峰"
        )
    if comparison_fragments:
        lines.append("本次 selected cases 的峰数对比为：" + "；".join(comparison_fragments) + "。")
    lines.append(
        "因此，较稳健的说法是：几何分类和壳穿越结构由 metric/junction 决定；峰高、相对亮度、弱峰是否越过检测阈值会随 emissivity 改变。报告只把两种 emissivity 都支持的峰数或峰位置趋势称为鲁棒。"
    )
    lines.append("")

    lines.append("## 参考文献")
    lines.append("")
    lines.append(
        "本文的静态壳解释框架参考 `ref/Imaging Signatures of the Israel Junction Photon Ring Evolution in Dynamical Thin Shell.pdf`。该文在 Schwarzschild-Schwarzschild junction 中识别出红移 cusp、V 形 transfer function 和 photon sphere/photon ring 非一一对应三类静态信号；本文只在现有 RN/RN-dS 输出直接支持的范围内延伸这些概念。"
    )
    lines.append("")

    lines.append("## 局限与后续工作")
    lines.append("")
    lines.append(
        "第一，当前 manifest 的 phase map 只有 `rn_mass_shell` 和 `rnds_lambda_shell` 两张，不能覆盖高电荷、大 `Q_jump`、靠近 cosmological horizon 的细结构。第二，profile 虽然已经围绕临界半径做 focused sampling，但 `redshift_cusp_orders` 仍需要采样点精确命中壳半径，因而比肉眼读图更保守；`v_shape_orders` 也只是局部极值指标。第三，本 atlas 是静态壳，没有包含参考论文后半部分的 shell collapse、传播延迟、动态红移跳变、盘-壳相互作用、旋转、等离子体或偏振。"
    )
    lines.append(
        "后续工作应使用 paper/dense preset 扩大 RN 的 `(Q_minus,Q_plus)` 与 RN-dS 的 `(Lambda_minus,Lambda_plus)` 扫描，增加自适应 profile 采样，并把 cusp/V-shape 检测从“命中网格点”升级为基于曲线左右导数或分段拟合的诊断。"
    )
    lines.append("")

    lines.append("## 附录：所有产物索引")
    lines.append("")
    lines.append("### 图像索引")
    lines.append("")
    lines.append("| figure_name | report_path | kind | tag | source_case_dir |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.extend(_figure_index_rows(figure_index))
    lines.append("")
    lines.append("### 产物索引")
    lines.append("")
    lines.append("| artifact | 含义 |")
    lines.append("| --- | --- |")
    lines.append("| `manifest.json` | atlas 的总索引，记录参数、物理筛选、phase map 摘要、selected cases、reference artifacts 与 figure index。 |")
    lines.append("| `phase_maps/*.csv` | 每个网格点的 admitted/rejected、warning、壳面源、`q_jump`、`lambda_jump` 和 photon-sphere 分类。 |")
    lines.append("| `phase_maps/*.png` / `figures/fig_*_phase.png` | phase map 可视化；`figures/` 下是报告用稳定文件名。 |")
    lines.append("| `cases/<case_id>/diagnostics.json` | 单个代表 case 的参数、标签、主 profile 诊断、backend comparison 摘要和 case figure index。 |")
    lines.append("| `cases/<case_id>/profile_*.(json/csv/png)` | 不同 emissivity 下的径向 profile 数据和图。JSON 包含每个采样点的壳穿越次数、盘交点、红移因子等。 |")
    lines.append("| `cases/<case_id>/image_*.png` | 由径向 profile 旋转生成的轴对称图像。 |")
    lines.append("| `cases/<case_id>/transfer_redshift.png` | 主 emissivity 下的 transfer function 与 redshift 诊断图。 |")
    if _merge_reference_artifacts(manifest, reference_manifest):
        lines.append("| `schwarzschild_reference/*` | Schwarzschild-Schwarzschild baseline 复现，包含参考论文 Fig. 3 到 Fig. 8 的 profile、图像和 sweep JSON。 |")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = _parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    reference_manifest_path = Path(args.reference_manifest).resolve() if args.reference_manifest else None
    manifest = _load_manifest(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_report(
            manifest,
            manifest_path=manifest_path,
            output_path=output_path,
            reference_manifest_path=reference_manifest_path,
        )
    )


if __name__ == "__main__":
    main()
