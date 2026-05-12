def test_package_imports():
    import spherical_raytracing

    assert spherical_raytracing.__all__


def test_public_api_exports_core_types():
    from spherical_raytracing import (
        DiskWindow,
        FiniteStaticObserver,
        HamiltonianTrajectorySolver,
        InfinityObserver,
        JunctionRayResult,
        LQGMetric,
        OdeTrajectorySolver,
        QuadTransferSolver,
        ReissnerNordstromDeSitterMetric,
        ReissnerNordstromMetric,
        SchwarzschildMetric,
        StaticJunctionHamiltonianSolver,
        StaticJunctionTransferSolver,
        StaticJunctionSpacetime,
        StaticShell,
        ThinDiskSource,
        compare_junction_results,
        trace_with_backend_compare,
    )

    assert SchwarzschildMetric
    assert LQGMetric
    assert ReissnerNordstromMetric
    assert ReissnerNordstromDeSitterMetric
    assert StaticShell
    assert StaticJunctionSpacetime
    assert JunctionRayResult
    assert InfinityObserver
    assert FiniteStaticObserver
    assert QuadTransferSolver
    assert StaticJunctionHamiltonianSolver
    assert StaticJunctionTransferSolver
    assert OdeTrajectorySolver
    assert HamiltonianTrajectorySolver
    assert ThinDiskSource
    assert DiskWindow
    assert compare_junction_results
    assert trace_with_backend_compare
