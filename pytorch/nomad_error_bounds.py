"""Certified local errors for two consecutive NomadSolver steps.

The solver supplies the binary64 results. An independent rational evaluation of
the same input field supplies the enclosure, including numerical roundoff.
See whitepaper/NOMAD_error_bounds.pdf for the theorem and the box criterion.
"""
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite, prod

import torch

if __package__:
    from .nomad_solver import NomadSolver
else:
    from nomad_solver import NomadSolver


@dataclass(frozen=True)
class ComponentCertificate:
    """All values are exact rationals. Errors have sign exact endpoint minus output."""
    low_output: Fraction
    high_output: Fraction
    exact_high_taylor: Fraction
    remainder_radius: Fraction

    @property
    def observed_increment(self):
        return self.high_output - self.low_output

    @property
    def high_roundoff_correction(self):
        return self.exact_high_taylor - self.high_output

    @property
    def signed_low_error(self):
        center = self.observed_increment + self.high_roundoff_correction
        return center - self.remainder_radius, center + self.remainder_radius

    @property
    def signed_high_error(self):
        center = self.high_roundoff_correction
        return center - self.remainder_radius, center + self.remainder_radius

    @property
    def absolute_low_error(self):
        lo, hi = self.signed_low_error
        return max(Fraction(0), lo, -hi), max(abs(lo), abs(hi))


def _collect(terms):
    collected = defaultdict(Fraction)
    for exponent, coefficient in terms:
        collected[exponent] += coefficient
    return {u: c for u, c in collected.items() if c}


def _next_terms(polynomial, field, order):
    for exponent, coefficient in polynomial.items():
        for j, power in enumerate(exponent):
            if power:
                for field_exp, field_coeff in field[j].items():
                    target = tuple(q + k - (i == j)
                                   for i, (q, k) in enumerate(zip(exponent, field_exp)))
                    yield target, coefficient * power * field_coeff / order


def _evaluate(polynomial, point, *, absolute=False):
    return sum(((abs(c) if absolute else c) * prod(x ** q for x, q in zip(point, u))
                for u, c in polynomial.items()), Fraction(0))


def _finite_doubles(tensor):
    return tensor.dtype == torch.float64 and bool(torch.isfinite(tensor).all())


def _step_values(a, k, z, h, order):
    solver = NomadSolver(a, k, n=order, device='cpu', coefficient_keep_threshold=0)
    values = solver.step(z[:, None, None], h)[:, 0, 0].tolist()
    if not all(isfinite(v) for v in values):
        raise ArithmeticError("NOMAD produced a nonfinite endpoint")
    return tuple(map(Fraction, values))


@torch.no_grad()
def certify_step(a, k, z, h, p, radii, *, high_order=None):
    """Return one ComponentCertificate per coordinate for a single forward step.

    a and k have shape (dimension, columns), z and radii have shape (dimension,).
    Supply finite float64 tensors for a, z, radii, nonnegative integral exponents,
    a positive finite binary64 step, and integer p >= 1. The higher order defaults
    to p+1. Set high_order > p to use a longer block of terms. The represented inputs
    specify the exact IVP. The box test raises ValueError if it cannot certify
    the step. The independent rational calculation is intended for modest orders.
    """
    if type(p) is not int or p < 1:
        raise ValueError("p must be a positive integer")
    if high_order is None:
        high_order = p + 1
    if type(high_order) is not int or high_order <= p:
        raise ValueError("high_order must be an integer greater than p")
    if type(h) is not float or not isfinite(h) or h <= 0:
        raise ValueError("h must be a positive finite float")
    if a.ndim != 2 or a.shape != k.shape or a.shape[0] == 0:
        raise ValueError("a and k must have matching nonempty row dimensions")
    if z.shape != (a.shape[0],) or radii.shape != z.shape:
        raise ValueError("z and radii must have one entry per coordinate")
    a, k, z, radii = (v.detach().cpu() for v in (a, k, z, radii))
    if not all(_finite_doubles(v) for v in (a, z, radii)) or not (radii > 0).all():
        raise ValueError("a, z and radii must be finite float64, with positive radii")
    if k.is_complex() or k.dtype == torch.bool:
        raise ValueError("exponents must be real integer values")
    columns = k.T.tolist()
    if any(not isfinite(v) or v < 0 or int(v) != v for column in columns for v in column):
        raise ValueError("negative, nonfinite or fractional exponents are outside the polynomial model")
    exponents = [tuple(map(int, column)) for column in columns]
    field = [_collect(zip(exponents, map(Fraction, row))) for row in a.tolist()]
    start = tuple(map(Fraction, z.tolist()))
    radius = tuple(map(Fraction, radii.tolist()))
    rho = tuple(abs(x) + r for x, r in zip(start, radius))
    dt = Fraction(h)
    for j, polynomial in enumerate(field):
        speed = _evaluate(polynomial, rho, absolute=True)
        if dt * speed >= radius[j]:
            raise ValueError(f"box test failed in coordinate {j}: h * speed >= radius")

    # C1 = F. Differentiate and divide by the new order at every subsequent stage.
    current = field
    high_taylor = list(start)
    for order in range(1, high_order + 1):
        for j, polynomial in enumerate(current):
            high_taylor[j] += dt ** order * _evaluate(polynomial, start)
        current = [_collect(_next_terms(poly, field, order + 1)) for poly in current]
    remainder = [dt ** (high_order + 1) * _evaluate(poly, rho, absolute=True) for poly in current]

    # Construct the two ordinary solver outputs, including all of their rounding.
    solver_exponents = k.to(torch.float64)
    low = _step_values(a, solver_exponents, z, h, p)
    high = _step_values(a, solver_exponents, z, h, high_order)
    return tuple(ComponentCertificate(*values)
                 for values in zip(low, high, high_taylor, remainder))


if __name__ == "__main__":
    # Reproduce the rational example checked in Nomad.Example.quadratic_certificate.
    a = torch.tensor([[1.0]], dtype=torch.float64)
    k = torch.tensor([[2]])
    z = torch.tensor([1.0], dtype=torch.float64)
    r = torch.tensor([0.5], dtype=torch.float64)
    certificate, = certify_step(a, k, z, 0.125, 2, r)
    print(f"Order 2 output = {certificate.low_output}")
    print(f"Order 3 output = {certificate.high_output}")
    print(f"Observed increment = {certificate.observed_increment}")
    print(f"Remainder radius = {certificate.remainder_radius}")
    lower, upper = certificate.signed_low_error
    print(f"Signed order 2 error in [{lower}, {upper}]")
