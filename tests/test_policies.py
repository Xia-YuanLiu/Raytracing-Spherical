from spherical_raytracing.diagnostics import EventType
from spherical_raytracing.policies import OutsideTracePolicy, SolverOptions, ThroughTracePolicy


def test_outside_policy_terminates_on_horizon():
    policy = OutsideTracePolicy()

    assert policy.is_terminal(EventType.HORIZON, "external") is True
    assert policy.is_terminal(EventType.TURNING_POINT, "external") is False
    assert policy.next_region(EventType.HORIZON, "external") is None


def test_through_policy_requires_metric_continuation():
    policy = ThroughTracePolicy(region_transitions={("external", EventType.HORIZON): "companion"})

    assert policy.is_terminal(EventType.HORIZON, "external") is False
    assert policy.next_region(EventType.HORIZON, "external") == "companion"


def test_through_policy_without_declared_transition_is_terminal():
    policy = ThroughTracePolicy()

    assert policy.is_terminal(EventType.HORIZON, "external") is True
    assert policy.next_region(EventType.HORIZON, "external") is None


def test_solver_options_defaults_are_conservative():
    options = SolverOptions()

    assert options.max_phi > 0.0
    assert options.critical_exclusion > 0.0
    assert options.root_atol > 0.0
