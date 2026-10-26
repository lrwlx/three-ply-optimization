"""Minimal regression tests for the core model and exact solver."""

from __future__ import annotations

import math
import unittest

from three_ply import MATERIALS, common_cofactors, evaluate, solve_extrema


class ThreePlyTests(unittest.TestCase):
    def test_common_cofactor_identity(self) -> None:
        material = MATERIALS["M0"]
        p, q, F1, F2, F3 = common_cofactors(material)
        values = {p: 0.25, q: -0.6}
        response = evaluate(
            material,
            math.degrees(math.atan(values[p])),
            math.degrees(math.atan(values[q])),
        )
        self.assertAlmostEqual(
            float((F2 / F1).subs(values)), -response["nu12"], places=12
        )
        self.assertAlmostEqual(
            float((F3 / F1).subs(values)), -response["nu13"], places=12
        )

    def test_reported_through_thickness_minimum(self) -> None:
        result = solve_extrema(MATERIALS["M0"], "nu13")
        self.assertEqual(result["candidate_count"], 8)
        self.assertAlmostEqual(result["theta1_deg"], 19.3296942976, places=8)
        self.assertAlmostEqual(result["theta2_deg"], -30.7017290644, places=8)
        self.assertAlmostEqual(result["objective_value"], -0.3128118051, places=9)


if __name__ == "__main__":
    unittest.main()

