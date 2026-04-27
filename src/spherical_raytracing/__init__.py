"""Core tools for ray tracing in static spherically symmetric spacetimes."""

from .diagnostics import CriticalCurve, EventType, RayDiagnostics, RayEvent, RayResult, RaySegment
from .imaging import ImageGrid, RadialProfile, RenderedImage, SamplingOptions, render_axisymmetric_image, sample_radial_profile
from .metrics import LQGMetric, SchwarzschildMetric
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
    "LQGMetric",
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
    "SchwarzschildMetric",
    "SolverOptions",
    "ThinDiskSource",
    "ThroughTracePolicy",
    "compute_intersections",
    "observed_intensity",
    "render_axisymmetric_image",
    "sample_radial_profile",
]
