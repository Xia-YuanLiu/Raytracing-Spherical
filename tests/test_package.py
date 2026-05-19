def test_package_imports():
    import spherical_raytracing

    assert spherical_raytracing.__all__


def test_public_api_exports_core_types():
    from spherical_raytracing import (
        CanonicalPayload,
        DiskWindow,
        FiniteStaticObserver,
        HamiltonianTrajectorySolver,
        InfinityObserver,
        JunctionRayResult,
        LQGMetric,
        LiDMBHMetadata,
        OdeTrajectorySolver,
        QuadTransferSolver,
        ReissnerNordstromDeSitterMetric,
        ReissnerNordstromMetric,
        SchwarzschildMetric,
        StaticDomainSelector,
        StaticJunctionHamiltonianSolver,
        StaticJunctionTransferSolver,
        StaticJunctionSpacetime,
        StaticShell,
        TabulatedMetric,
        ThinDiskSource,
        ValidationGate,
        ValidationOptions,
        compare_junction_results,
        load_metric_npz,
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
    assert TabulatedMetric
    assert load_metric_npz
    assert CanonicalPayload
    assert LiDMBHMetadata
    assert ValidationOptions
    assert StaticDomainSelector
    assert ValidationGate
