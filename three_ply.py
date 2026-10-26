"""Core solver for Poisson-ratio extrema in symmetric [theta1, theta2, theta1] plies.

The implementation follows the common-cofactor formulation used in the
associated manuscript. It contains only the material model, classical-laminate
response, symbolic resultant construction and numerical root isolation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class Material:
    """Special-orthotropic material card using the manuscript notation."""

    name: str
    E11: float
    E22: float
    E33: float
    G12: float
    G13: float
    G23: float
    nu12: float
    nu13: float
    nu23: float

    def compliance4(self) -> np.ndarray:
        """Return S for strain (e11, e22, e33, g12)."""
        S = np.zeros((4, 4), dtype=float)
        S[0, 0] = 1.0 / self.E11
        S[0, 1] = S[1, 0] = -self.nu12 / self.E11
        S[0, 2] = S[2, 0] = -self.nu13 / self.E11
        S[1, 1] = 1.0 / self.E22
        S[1, 2] = S[2, 1] = -self.nu23 / self.E22
        S[2, 2] = 1.0 / self.E33
        S[3, 3] = 1.0 / self.G12
        return S

    def stiffness4(self) -> np.ndarray:
        return np.linalg.inv(self.compliance4())


MATERIALS = {
    "M0": Material("IM7/977-3", 159.0, 9.20, 9.20, 4.37, 4.37, 2.57,
                   0.253, 0.253, 0.45),
    "M1": Material("IM7/8552", 171.4, 9.08, 9.08, 5.29, 5.29, 2.80,
                   0.32, 0.32, 0.50),
    "M2": Material("T700/M21", 130.0, 7.70, 7.70, 4.80, 4.80, 3.80,
                   0.30, 0.30, 0.35),
    "M3": Material("IM6/3501-6", 144.7, 9.65, 9.65, 5.20, 5.20, 3.40,
                   0.30, 0.30, 0.45),
    "M4": Material("T300/5208", 132.4, 10.76, 10.76, 5.65, 5.65, 3.38,
                   0.24, 0.24, 0.49),
}


def rotated_stiffness(C: np.ndarray, theta_deg: float) -> np.ndarray:
    """Rotate the reduced 4x4 stiffness into the laminate axes."""
    t = np.deg2rad(theta_deg)
    c, s = math.cos(t), math.sin(t)
    c2, s2, cs = c * c, s * s, c * s

    C11, C12, C22 = C[0, 0], C[0, 1], C[1, 1]
    C13, C23, C33, C66 = C[0, 2], C[1, 2], C[2, 2], C[3, 3]

    out = np.empty((4, 4), dtype=float)
    out[0, 0] = C11 * c2**2 + (2 * C12 + 4 * C66) * c2 * s2 + C22 * s2**2
    out[1, 1] = C22 * c2**2 + (2 * C12 + 4 * C66) * c2 * s2 + C11 * s2**2
    out[0, 1] = out[1, 0] = (
        C12 * (c2**2 + s2**2) + (C11 + C22 - 4 * C66) * c2 * s2
    )
    out[2, 2] = C33
    out[3, 3] = (
        C66 * (c2**2 + s2**2)
        + (C11 + C22 - 2 * C12 - 2 * C66) * c2 * s2
    )
    out[0, 2] = out[2, 0] = C13 * c2 + C23 * s2
    out[1, 2] = out[2, 1] = C23 * c2 + C13 * s2
    out[0, 3] = out[3, 0] = (
        (C11 - C12 - 2 * C66) * c2 * cs
        + (C12 - C22 + 2 * C66) * c * s2 * s
    )
    out[1, 3] = out[3, 1] = (
        (C11 - C12 - 2 * C66) * c * s2 * s
        + (C12 - C22 + 2 * C66) * c2 * cs
    )
    out[2, 3] = out[3, 2] = (C13 - C23) * cs
    return out


def three_ply_A(
    material: Material, theta1_deg: float, theta2_deg: float
) -> np.ndarray:
    """Normalized A for equal-thickness [theta1, theta2, theta1]."""
    C = material.stiffness4()
    return (
        2 * rotated_stiffness(C, theta1_deg)
        + rotated_stiffness(C, theta2_deg)
    ) / 3


def evaluate(material: Material, theta1_deg: float, theta2_deg: float) -> dict:
    """Evaluate both Poisson ratios and the inverse extensional stiffness."""
    A = three_ply_A(material, theta1_deg, theta2_deg)
    J = np.linalg.inv(A)
    return {
        "nu12": float(-J[1, 0] / J[0, 0]),
        "nu13": float(-J[2, 0] / J[0, 0]),
        "A": A,
        "J": J,
    }


def _rational(value: float) -> sp.Rational:
    return sp.Rational(str(value))


def common_cofactors(material: Material):
    """Construct F1, F2 and F3 in tangent coordinates.

    The returned polynomials satisfy nu12=-F2/F1 and nu13=-F3/F1 after the
    common tangent-coordinate factor is cancelled exactly.
    """
    p, q = sp.symbols("p q", real=True)
    eta, zeta = 1 + p**2, 1 + q**2

    S = sp.Matrix(
        [
            [1 / _rational(material.E11), -_rational(material.nu12) / _rational(material.E11),
             -_rational(material.nu13) / _rational(material.E11), 0],
            [-_rational(material.nu12) / _rational(material.E11), 1 / _rational(material.E22),
             -_rational(material.nu23) / _rational(material.E22), 0],
            [-_rational(material.nu13) / _rational(material.E11),
             -_rational(material.nu23) / _rational(material.E22),
             1 / _rational(material.E33), 0],
            [0, 0, 0, 1 / _rational(material.G12)],
        ]
    )
    C = S.inv()
    C11, C12, C22 = C[0, 0], C[0, 1], C[1, 1]
    C13, C23, C33, C66 = C[0, 2], C[1, 2], C[2, 2], C[3, 3]

    quartic = lambda a, b, c, t: a + b * t**2 + c * t**4
    quadratic = lambda a, b, t: a + b * t**2
    shear = lambda a, b, t: t * (a + b * t**2)

    specs = {
        (0, 0): (lambda t: quartic(C11, 2 * C12 + 4 * C66, C22, t), 2),
        (1, 1): (lambda t: quartic(C22, 2 * C12 + 4 * C66, C11, t), 2),
        (3, 3): (lambda t: quartic(C66, C11 + C22 - 2 * C12 - 2 * C66, C66, t), 2),
        (0, 1): (lambda t: quartic(C12, C11 + C22 - 4 * C66, C12, t), 2),
        (0, 2): (lambda t: quadratic(C13, C23, t), 1),
        (1, 2): (lambda t: quadratic(C23, C13, t), 1),
        (0, 3): (lambda t: shear(C11 - C12 - 2 * C66, C12 - C22 + 2 * C66, t), 2),
        (1, 3): (lambda t: shear(C12 - C22 + 2 * C66, C11 - C12 - 2 * C66, t), 2),
        (2, 3): (lambda t: (C13 - C23) * t, 1),
    }

    at_p = {key: spec[0](p) for key, spec in specs.items()}
    at_q = {key: spec[0](q) for key, spec in specs.items()}
    B = sp.zeros(4)
    B[2, 2] = 3 * C33 * eta**2 * zeta**2
    for (i, j), (_, power) in specs.items():
        if power == 2:
            B[i, j] = 2 * at_p[(i, j)] * zeta**2 + at_q[(i, j)] * eta**2
        else:
            B[i, j] = (
                2 * at_p[(i, j)] * eta * zeta**2
                + at_q[(i, j)] * zeta * eta**2
            )
        if i != j:
            B[j, i] = B[i, j]

    common = eta**4 * zeta**4

    def reduce(expression):
        result = sp.cancel(sp.factor(expression / common))
        if sp.denom(result) != 1:
            raise RuntimeError("Common tangent-coordinate factor did not cancel.")
        return sp.expand(result)

    return p, q, reduce(B.cofactor(0, 0)), reduce(B.cofactor(0, 1)), reduce(
        B.cofactor(0, 2)
    )


def _coefficient_matrix(expression, p, q) -> np.ndarray:
    poly = sp.Poly(expression, p, q)
    matrix = np.zeros((poly.degree(p) + 1, poly.degree(q) + 1), dtype=float)
    for (degree_p, degree_q), coefficient in poly.terms():
        matrix[degree_p, degree_q] = float(sp.N(coefficient, 40))
    return matrix


def _real_roots(coefficients, tolerance: float = 1e-7) -> list[float]:
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.size < 2:
        return []
    scale = float(np.max(np.abs(coefficients)))
    if not np.isfinite(scale) or scale == 0:
        return []
    roots = np.roots(coefficients / scale)
    return [
        float(root.real)
        for root in roots
        if abs(root.imag) <= tolerance * max(1.0, abs(root.real))
    ]


def _gradient(material: Material, theta1: float, theta2: float, target: str) -> np.ndarray:
    step = 1e-5
    return np.array(
        [
            (
                evaluate(material, theta1 + step, theta2)[target]
                - evaluate(material, theta1 - step, theta2)[target]
            )
            / (2 * step),
            (
                evaluate(material, theta1, theta2 + step)[target]
                - evaluate(material, theta1, theta2 - step)[target]
            )
            / (2 * step),
        ]
    )


def solve_extrema(material: Material, target: str) -> dict:
    """Solve one objective by resultant elimination and boundary checks."""
    if target not in ("nu12", "nu13"):
        raise ValueError("target must be 'nu12' or 'nu13'")

    started = time.time()
    p, q, F1, F2, F3 = common_cofactors(material)
    objective = F2 if target == "nu12" else F3

    Gp = sp.expand(sp.diff(objective, p) * F1 - objective * sp.diff(F1, p))
    Gq = sp.expand(sp.diff(objective, q) * F1 - objective * sp.diff(F1, q))
    resultant = sp.factor(
        sp.resultant(sp.Poly(Gp, p, q), sp.Poly(Gq, p, q), q)
    )
    gp_matrix = _coefficient_matrix(Gp, p, q)

    candidate_p = []
    factor_rows = []
    for factor, multiplicity in sp.factor_list(resultant, p)[1]:
        if factor == p:
            continue
        factor_p = sp.Poly(factor, p)
        if sp.expand(factor.subs(p, -p) - factor) != 0:
            polynomial = factor_p
            roots = _real_roots(
                [float(sp.N(c / polynomial.LC(), 40)) for c in polynomial.all_coeffs()]
            )
            candidate_p.extend(root for root in roots if root > 1e-12)
            u_degree = None
        else:
            u = sp.symbols("u")
            polynomial = sp.Poly(factor.subs(p, sp.sqrt(u)), u)
            roots = _real_roots(
                [float(sp.N(c / polynomial.LC(), 40)) for c in polynomial.all_coeffs()]
            )
            candidate_p.extend(math.sqrt(root) for root in roots if root > 1e-12)
            u_degree = int(polynomial.degree())
        factor_rows.append(
            {
                "p_degree": int(factor_p.degree()),
                "u_degree": u_degree,
                "multiplicity": int(multiplicity),
            }
        )

    unique_p = []
    for value in sorted(candidate_p):
        if not unique_p or abs(value - unique_p[-1]) > 1e-7 * max(1.0, abs(value)):
            unique_p.append(value)

    candidates = []
    for p_value in unique_p:
        p_powers = np.array([p_value**i for i in range(gp_matrix.shape[0])])
        q_coefficients = (p_powers @ gp_matrix)[::-1]
        for q_value in _real_roots(q_coefficients):
            if not np.isfinite(q_value) or abs(q_value) > 1e8:
                continue
            theta1 = math.degrees(math.atan(p_value))
            theta2 = math.degrees(math.atan(q_value))
            if not (-90 < theta1 < 90 and -90 < theta2 < 90):
                continue
            if np.linalg.norm(_gradient(material, theta1, theta2, target)) > 1e-6:
                continue
            response = evaluate(material, theta1, theta2)
            if not math.isfinite(response[target]):
                continue
            if any(
                abs(theta1 - old["theta1"]) < 1e-6
                and abs(theta2 - old["theta2"]) < 1e-6
                for old in candidates
            ):
                continue
            candidates.append(
                {
                    "theta1": theta1,
                    "theta2": theta2,
                    "nu12": response["nu12"],
                    "nu13": response["nu13"],
                }
            )

    if not candidates:
        raise RuntimeError(f"No finite interior stationary candidate for {target}.")

    symmetry_pool = [row for row in candidates if row["theta1"] > 0]
    best = min(symmetry_pool or candidates, key=lambda row: row[target])

    boundary_angles = np.linspace(-90.0, 90.0, 3601)
    boundary_values = []
    for fixed in (-90.0, 0.0, 90.0):
        boundary_values.extend(
            evaluate(material, fixed, theta)[target] for theta in boundary_angles
        )
        boundary_values.extend(
            evaluate(material, theta, fixed)[target] for theta in boundary_angles
        )

    return {
        "material": material.name,
        "target": target,
        "theta1_deg": best["theta1"],
        "theta2_deg": best["theta2"],
        "objective_value": best[target],
        "cross_value": best["nu13" if target == "nu12" else "nu12"],
        "boundary_minimum": float(min(boundary_values)),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "resultant_degree": int(sp.Poly(resultant, p).degree()),
        "resultant_factors": factor_rows,
        "elapsed_seconds": time.time() - started,
    }

