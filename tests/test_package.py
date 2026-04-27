def test_package_imports():
    import spherical_raytracing

    assert spherical_raytracing.__all__


def test_public_api_exports_core_types():
    from spherical_raytracing import (
        DiskWindow,
        FiniteStaticObserver,
        HamiltonianTrajectorySolver,
        InfinityObserver,
        LQGMetric,
        OdeTrajectorySolver,
        QuadTransferSolver,
        SchwarzschildMetric,
        ThinDiskSource,
    )

    assert SchwarzschildMetric
    assert LQGMetric
    assert InfinityObserver
    assert FiniteStaticObserver
    assert QuadTransferSolver
    assert OdeTrajectorySolver
    assert HamiltonianTrajectorySolver
    assert ThinDiskSource
    assert DiskWindow
