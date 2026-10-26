# Common-cofactor resultant solver for three-ply laminates

This repository contains the core implementation for:

**A common-cofactor resultant method for in-plane and through-thickness Poisson-ratio extrema in symmetric three-ply laminates**

The code studies the equal-thickness, special-orthotropic layup `[theta1, theta2, theta1]`. It keeps the essential method and omits plotting, random scans, manuscript assembly, RVE meshing and Abaqus post-processing.

## Method

For the normalized extensional stiffness `A` and its inverse `J = A^-1`, the two Poisson ratios are

```text
nu12 = -J21 / J11
nu13 = -J31 / J11.
```

The inverse entries are replaced by cofactors of `A`:

```text
nu12 = -Gamma21(A) / Gamma11(A)
nu13 = -Gamma31(A) / Gamma11(A).
```

This gives one shared denominator and two objective-specific numerators. With `p = tan(theta1)` and `q = tan(theta2)`, the stiffness entries become rational functions. After clearing denominators and cancelling the common tangent-coordinate factor, the solver obtains `F1`, `F2` and `F3`, so that

```text
nu12 = -F2/F1,    nu13 = -F3/F1.
```

For each objective, the interior stationary equations are

```text
Gp = (dF/dp) F1 - F (dF1/dp) = 0
Gq = (dF/dq) F1 - F (dF1/dq) = 0.
```

The code eliminates `q` with a symbolic resultant, isolates real roots of the resulting univariate polynomial, and rejects roots that fail the original stationary equations. Finite edge families are checked separately.

## Files

| File | Purpose |
| --- | --- |
| `three_ply.py` | Material model, laminate rotation, common-cofactor construction and solver |
| `example.py` | Command-line example for IM7/977-3 |
| `test_three_ply.py` | Cofactor identity and reported through-thickness minimum |
| `requirements.txt` | NumPy and SymPy dependencies |

## Run

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Run the in-plane example:

```bash
python example.py --target nu12
```

Run both objective-specific examples:

```bash
python example.py --target both
```

A single objective takes roughly 20 seconds on a typical desktop because the common-cofactor algebra is constructed exactly with SymPy.

Run the regression tests:

```bash
python -m unittest -v
```

## Reference result

For IM7/977-3, the two reported minima are:

| Objective | Layup `[theta1, theta2, theta1]` | Minimum | Other ratio |
| --- | --- | ---: | ---: |
| `nu12` | `[12.355924, 63.616866, 12.355924] deg` | `-0.431265891` | `0.624531032` |
| `nu13` | `[19.329694, -30.701729, 19.329694] deg` | `-0.312811805` | `1.582055826` |

## Scope

The implementation applies to equal-thickness plies in the material class stated in the manuscript. The resultant construction is symbolic, while root isolation and residual checks are numerical. Boundary cases are checked by a separate numerical scan.

Associated manuscript repository: <https://github.com/lrwlx/three-ply-optimization>
