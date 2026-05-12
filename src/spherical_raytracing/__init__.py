"""Core tools for ray tracing in static spherically symmetric spacetimes."""

from .diagnostics import CriticalCurve, EventType, RayDiagnostics, RayEvent, RayResult, RaySegment
from .imaging import ImageGrid, RadialProfile, RenderedImage, SamplingOptions, render_axisymmetric_image, sample_radial_profile
from .junctions import (
    JunctionPhysicsDiagnostics,
    JunctionRayResult,
    SegmentConstants,
    ShellCrossingRecord,
    StaticJunctionSpacetime,
    StaticShell,
    annotate_junction_intersections,
    compare_junction_results,
    junction_redshift_factor,
    match_static_shell,
)
from .junction_tracing import StaticJunctionHamiltonianSolver, StaticJunctionTransferSolver, trace_with_backend_compare
from .metrics import LQGMetric, ReissnerNordstromDeSitterMetric, ReissnerNordstromMetric, SchwarzschildMetric
from .observers import FiniteStaticObserver, InfinityObserver
from .policies import OutsideTracePolicy, SolverOptions, ThroughTracePolicy
from .solvers import HamiltonianTrajectorySolver, OdeTrajectorySolver, QuadTransferSolver
from .sources import ObservedIntensity, ThinDiskSource, observed_intensity
from .transfer import DiskWindow, IntersectionRecord, compute_intersections

__all__ = [
    "CriticalCurve",
    "DiskWindow",
    "EventType",
    "FiniteStaticObserver",
    "HamiltonianTrajectorySolver",
    "ImageGrid",
    "InfinityObserver",
    "IntersectionRecord",
    "JunctionPhysicsDiagnostics",
    "JunctionRayResult",
    "LQGMetric",
    "SegmentConstants",
    "ShellCrossingRecord",
    "ObservedIntensity",
    "OdeTrajectorySolver",
    "OutsideTracePolicy",
    "QuadTransferSolver",
    "RadialProfile",
    "RayDiagnostics",
    "RayEvent",
    "RayResult",
    "RaySegment",
    "RenderedImage",
    "SamplingOptions",
    "StaticJunctionHamiltonianSolver",
    "StaticJunctionTransferSolver",
    "StaticJunctionSpacetime",
    "StaticShell",
    "ReissnerNordstromDeSitterMetric",
    "ReissnerNordstromMetric",
    "SchwarzschildMetric",
    "SolverOptions",
    "ThinDiskSource",
    "annotate_junction_intersections",
    "compare_junction_results",
    "junction_redshift_factor",
    "match_static_shell",
    "trace_with_backend_compare",
    "ThroughTracePolicy",
    "compute_intersections",
    "observed_intensity",
    "render_axisymmetric_image",
    "sample_radial_profile",
]
