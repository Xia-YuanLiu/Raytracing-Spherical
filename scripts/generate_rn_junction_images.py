from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_static_junction_profiles import build_profile, write_image
from spherical_raytracing.metrics import ReissnerNordstromMetric


@dataclass(frozen=True)
class RNCase:
    name: str
    description: str
    m_minus: float
    m_plus: float
    q_minus: float
    q_plus: float
    shell_radius: float
    observer_radius: float = 50.0


RN_CASES: tuple[RNCase, ...] = (
    RNCase(
        name="small_charge_schwarzschild_limit",
        description="Small equal charges; baseline close to the Schwarzschild junction image.",
        m_minus=0.10,
        m_plus=1.00,
        q_minus=0.02,
        q_plus=0.02,
        shell_radius=3.10,
    ),
    RNCase(
        name="equal_moderate_charge",
        description="Equal moderate charges isolate charge effects without a surface charge jump.",
        m_minus=0.40,
        m_plus=1.00,
        q_minus=0.25,
        q_plus=0.25,
        shell_radius=3.40,
    ),
    RNCase(
        name="outer_surface_charge_dominant",
        description="Exterior charge exceeds interior charge, so the shell carries positive surface charge.",
        m_minus=0.35,
        m_plus=1.00,
        q_minus=0.05,
        q_plus=0.55,
        shell_radius=3.30,
    ),
    RNCase(
        name="inner_surface_charge_dominant",
        description="Interior charge exceeds exterior charge while both sides remain subextremal.",
        m_minus=0.70,
        m_plus=1.00,
        q_minus=0.62,
        q_plus=0.15,
        shell_radius=3.60,
    ),
    RNCase(
        name="near_extremal_outer_charge",
        description="Near-extremal exterior RN metric, changing the outer photon-sphere scale.",
        m_minus=0.45,
        m_plus=1.00,
        q_minus=0.20,
        q_plus=0.96,
        shell_radius=3.15,
    ),
    RNCase(
        name="near_extremal_inner_charge",
        description="Near-extremal interior RN metric probes strong inner redshift matching.",
        m_minus=0.85,
        m_plus=1.00,
        q_minus=0.80,
        q_plus=0.25,
        shell_radius=3.75,
    ),
    RNCase(
        name="shell_near_outer_photon_sphere",
        description="Shell just outside the outer unstable photon sphere, emphasizing shell-crossing features.",
        m_minus=0.20,
        m_plus=1.00,
        q_minus=0.10,
        q_plus=0.45,
        shell_radius=2.95,
    ),
    RNCase(
        name="large_shell_weak_matching",
        description="Larger shell radius weakens the junction redshift discontinuity.",
        m_minus=0.60,
        m_plus=1.00,
        q_minus=0.20,
        q_plus=0.20,
        shell_radius=8.00,
    ),
    RNCase(
        name="light_inner_mass_strong_contrast",
        description="Low interior mass with charged exterior gives a high mass-contrast junction.",
        m_minus=0.15,
        m_plus=1.00,
        q_minus=0.05,
        q_plus=0.70,
        shell_radius=3.25,
    ),
    RNCase(
        name="heavy_inner_mass_small_charge_jump",
        description="Interior mass close to exterior mass with a small charge discontinuity.",
        m_minus=0.90,
        m_plus=1.00,
        q_minus=0.30,
        q_plus=0.45,
        shell_radius=4.20,
    ),
)


def _parse_case_filter(values: list[str] | None) -> tuple[RNCase, ...]:
    if not values:
        return RN_CASES
    by_name = {case.name: case for case in RN_CASES}
    missing = [name for name in values if name not in by_name]
    if missing:
        raise SystemExit(f"unknown RN case(s): {', '.join(missing)}")
    return tuple(by_name[name] for name in values)


def _outer_photon_sphere_radius(case: RNCase) -> float:
    metric = ReissnerNordstromMetric(mass=case.m_plus, charge=case.q_plus, region="outer")
    photon_spheres = metric.photon_spheres()
    if not photon_spheres:
        return math.nan
    return photon_spheres[-1]


def _critical_impact_parameter(case: RNCase) -> float:
    metric = ReissnerNordstromMetric(mass=case.m_plus, charge=case.q_plus, region="outer")
    critical_curves = metric.critical_curves()
    if not critical_curves:
        return math.nan
    return critical_curves[-1].b_crit


def _profile_args(args: argparse.Namespace, case: RNCase) -> argparse.Namespace:
    return argparse.Namespace(
        m_minus=case.m_minus,
        m_plus=case.m_plus,
        q_minus=case.q_minus,
        q_plus=case.q_plus,
        lambda_minus=None,
        lambda_plus=None,
        shell_radius=case.shell_radius,
        observer_radius=case.observer_radius,
        compare_backends=args.compare_backends,
        disk_r_min=args.disk_r_min,
        disk_r_max=args.disk_r_max,
        max_order=args.max_order,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        samples=args.samples,
        max_refine=args.max_refine,
    )


def _validate_case(case: RNCase) -> None:
    if case.m_minus <= 0.0 or case.m_plus <= 0.0:
        raise SystemExit(f"{case.name}: masses must be positive")
    if abs(case.q_minus) >= case.m_minus:
        raise SystemExit(f"{case.name}: q_minus must be subextremal for this sweep")
    if abs(case.q_plus) >= case.m_plus:
        raise SystemExit(f"{case.name}: q_plus must be subextremal for this sweep")
    if case.shell_radius <= _outer_photon_sphere_radius(case):
        raise SystemExit(f"{case.name}: shell_radius must exceed the outer photon sphere")
    if case.observer_radius <= case.shell_radius:
        raise SystemExit(f"{case.name}: observer_radius must exceed shell_radius")
    inner = ReissnerNordstromMetric(mass=case.m_minus, charge=case.q_minus, region="inner")
    outer = ReissnerNordstromMetric(mass=case.m_plus, charge=case.q_plus, region="outer")
    if inner.A(case.shell_radius) <= 0.0 or outer.A(case.shell_radius) <= 0.0:
        raise SystemExit(f"{case.name}: shell must lie in a static region on both sides")
    if outer.A(case.observer_radius) <= 0.0:
        raise SystemExit(f"{case.name}: observer must lie in the outer static region")


def render_case(case: RNCase, args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    _validate_case(case)
    profile = build_profile("rn", _profile_args(args, case))
    image_name = f"{case.name}.png"
    profile_name = f"{case.name}.json"
    write_image(profile, output_dir / image_name, args.image_size)
    (output_dir / profile_name).write_text(json.dumps(profile, indent=2, sort_keys=True))
    return {
        "name": case.name,
        "description": case.description,
        "image": image_name,
        "profile": profile_name,
        "parameters": profile["parameters"],
        "warnings": profile["warnings"],
        "sample_count": len(profile["samples"]),
        "outer_photon_sphere_radius": _outer_photon_sphere_radius(case),
        "outer_critical_impact_parameter": _critical_impact_parameter(case),
        "sampling_diagnostics": profile["sampling_diagnostics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images for a curated RN static-junction parameter sweep.")
    parser.add_argument("--output-dir", required=True, help="Directory for manifest, per-case JSON, and PNG outputs")
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--max-refine", type=int, default=1)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--disk-r-min", type=float, default=0.25)
    parser.add_argument("--disk-r-max", type=float, default=100.0)
    parser.add_argument("--alpha-min", type=float, default=0.01)
    parser.add_argument("--alpha-max", type=float, default=0.25)
    parser.add_argument("--case", action="append", help="Case name to render; may be passed multiple times")
    parser.add_argument("--compare-backends", action="store_true")
    args = parser.parse_args()

    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.max_refine < 0:
        raise SystemExit("--max-refine must be non-negative")
    if args.max_order <= 0:
        raise SystemExit("--max-order must be positive")
    if args.image_size <= 0:
        raise SystemExit("--image-size must be positive")
    if args.disk_r_min <= 0.0 or args.disk_r_max <= args.disk_r_min:
        raise SystemExit("disk window must satisfy 0 < --disk-r-min < --disk-r-max")
    if args.alpha_min < 0.0 or args.alpha_max <= args.alpha_min:
        raise SystemExit("alpha window must satisfy 0 <= --alpha-min < --alpha-max")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = _parse_case_filter(args.case)
    rendered_cases = [render_case(case, args, output_dir) for case in cases]
    manifest = {
        "model": "rn_static_junction_sweep",
        "case_count": len(rendered_cases),
        "notes": [
            "All preset cases are subextremal RN/RN static junctions.",
            "Shell radii are outside the outer metric unstable photon sphere and static on both sides.",
            "charge_mismatch_surface_source means the junction shell carries surface charge; it is not a rejection.",
        ],
        "cases": rendered_cases,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
