"""Run one or both reported extrema for IM7/977-3."""

from __future__ import annotations

import argparse
import json

from three_ply import MATERIALS, solve_extrema


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=("nu12", "nu13", "both"),
        default="nu12",
        help="Poisson ratio to minimize; 'both' runs the two examples.",
    )
    args = parser.parse_args()

    targets = ("nu12", "nu13") if args.target == "both" else (args.target,)
    summary = []
    for target in targets:
        result = solve_extrema(MATERIALS["M0"], target)
        summary.append(
            {
                key: result[key]
                for key in (
                    "material",
                    "target",
                    "theta1_deg",
                    "theta2_deg",
                    "objective_value",
                    "cross_value",
                    "boundary_minimum",
                    "candidate_count",
                    "resultant_degree",
                )
            }
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

