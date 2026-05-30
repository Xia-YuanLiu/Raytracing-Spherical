# Physics And Numerical Conventions

Date: 2026-05-29

## Coordinates

Metrics use static, spherically symmetric coordinates with areal radius `r`:

```text
ds^2 = -A(r) dt^2 + B(r) dr^2 + r^2 dOmega^2
```

The radius `r` is the areal radius, so a constant-`r` sphere has area
`4 pi r^2`.

## Observers

`InfinityObserver` represents the asymptotically flat observer path for
analytical metrics that can support `u=1/r -> 0`.

`FiniteStaticObserver` represents a static observer at finite `r_obs`. It is the
required observer for `TabulatedMetric`, because finite tables do not define
metric values at infinity.

## Ray Coordinate

Single-metric solvers trace in `u=1/r` and polar angle `phi`. A `RaySegment`
stores its endpoint event and a `u_at(phi)` evaluator. That evaluator is part of
the public behavior because disk intersections call it to recover crossing
radii.

## Thin-Disk Redshift

For a single metric and finite static observer, the static gravitational
redshift factor used by the thin-disk helpers is:

```text
g = sqrt(A(r_emit) / A(r_obs))
```

Observed intensity uses redshift weights `g^4` for the static-source model.

## Static Shell Matching

`StaticJunctionSpacetime` currently supports same-family analytical static
junctions. At a shell radius `R`, matching keeps angular momentum `L` fixed and
rescales the backward traced energy by:

```text
E_after / E_before = sqrt(A_to(R) / A_from(R))
b_after = L / E_after
```

For identical inner and outer metrics, this degenerates to identity:
`E_after/E_before = 1` and `b_after = b_before`.

The junction redshift helper multiplies the local static redshift by the
forward shell-crossing energy ratios for shell crossings before the disk
emission angle. In the identical-metric degeneration that shell product is 1.

## Termination Events

Termination event semantics are public compatibility behavior. Important events
include:

- `horizon`: the ray reaches a metric horizon boundary.
- `black_hole_horizon` / `cosmological_horizon`: diagnostic termination reasons
  classifying which finite radial boundary was hit.
- `escape`: an outward ray reaches infinity for an unbounded analytical domain.
- `outer_boundary`: a finite non-horizon outer radial boundary is reached.
- `inner_boundary`: an inner non-horizon cutoff is reached.
- `turning_point`: an intermediate segment endpoint where radial motion reverses.
- `max_phi`: the configured angular tracing limit truncates the ray.
- `near_critical`: tracing is intentionally excluded near a reachable critical
  curve.

## Numerical Table Boundaries

`TabulatedMetric` interpolates `A(r)` and `C(r)=1/B(r)` with PCHIP over a chosen
static-domain subarray. Horizon zero anchors are injected only at static-domain
endpoints that match metadata horizons within the documented tolerance.

Queries outside the selected static domain or outside finite table support raise
instead of extrapolating. `TabulatedMetric + InfinityObserver`,
`TabulatedMetric + ThroughTracePolicy`, and `TabulatedMetric + Hamiltonian` are
not public supported combinations in the current policy.
