import math

from spherical_raytracing.metrics import SchwarzschildMetric
from spherical_raytracing.observers import InfinityObserver
from spherical_raytracing.solvers import QuadTransferSolver
from spherical_raytracing.transfer import DiskWindow, compute_intersections


def test_direct_intersection_for_large_b_is_near_b():
    metric = SchwarzschildMetric(mass=1.0)
    result = QuadTransferSolver(metric=metric, observer=InfinityObserver()).trace_b(30.0)

    intersections = compute_intersections(result, DiskWindow(r_min=6.0, r_max=200.0), max_order=1)

    assert len(intersections) == 1
    assert intersections[0].m == 1
    assert math.isclose(intersections[0].phi, math.pi / 2)
    assert math.isclose(intersections[0].r, 30.0, rel_tol=0.1)
    assert intersections[0].path_class == "direct"


def test_source_window_filters_out_small_disk():
    metric = SchwarzschildMetric(mass=1.0)
    result = QuadTransferSolver(metric=metric, observer=InfinityObserver()).trace_b(30.0)

    intersections = compute_intersections(result, DiskWindow(r_min=100.0, r_max=200.0), max_order=1)

    assert intersections == []
