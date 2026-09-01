# Octave-to-Python Solver Migration Slides

These slide notes focus on the adults time series for all 16 spatial cells.

Input comparison table:

`data/out/solver_side_by_side_adults_scipy_daily.csv`

---

## 1. Motivation

- The Octave implementation is slow and difficult to maintain.
- Python enables profiling, tests, backend comparison, and solver modernization.
- The migration decision must separate compatibility from numerical modernization.

---

## 2. Backend Summary

| Backend | Numerical policy | Role |
|---|---|---|
| Octave/reference | Fixed RK4 with log-correction | Historical reference |
| `legacy_optimized` | Same RK4/log-correction, faster | Compatibility candidate |
| `scipy_chunked` | Adaptive `solve_ivp`, daily forcing | Modernization candidate |

---

## 3. Comparison Method

- Variable: `adults`.
- NetCDF outputs are normalized to `(longitude, latitude, time)`.
- Dataset size: `16` spatial cells x `366` days.
- Relative difference: `abs(scipy_chunked - Octave) / (abs(Octave) + 1)`.
- The largest-divergence cell is detected from the data, not preselected.
- Stable active summary threshold: `abs(Octave adults) >= 1` and `< 1e+10`.

---

## 4. Adults Time Series For All 16 Cells

![Adults time series for 16 cells](../plots/solver_migration/adults_timeseries_16_cells_octave_vs_scipy_daily.png)

How to read this:

- Blue solid line: Octave/reference.
- Orange dashed line: `scipy_chunked`.
- Monthly gridlines show the complete year.
- The magenta-highlighted panel is the largest-divergence cell.

---

## 5. Relative Difference Over Time

![Relative difference for 16 cells](../plots/solver_migration/adults_relative_difference_16_cells.png)

How to read this:

- Values near zero indicate close agreement.
- Persistent large values indicate trajectory-level divergence.
- The same largest-divergence cell is highlighted.

---

## 6. Zoom: Largest-Divergence Cell

- Longitude: `10`
- Latitude: `46`
- Date of largest relative difference: `2024-12-23`
- Largest relative difference: `1.000e+00`

![Largest-divergence cell](../plots/solver_migration/known_divergence_cell_timeseries.png)

Interpretation:

- Octave/reference and `legacy_optimized` follow the RK4/log-correction trajectory.
- `scipy_chunked` remains bounded.
- This is a solver-policy difference, not a file-orientation issue.

---

## 7. Numerical Summary

Stable active values compared: `3515`

| Metric | scipy_chunked vs Octave/reference |
|---|---:|
| Mean relative difference | 5.691e-04 |
| Median relative difference | 2.184e-06 |
| P95 relative difference | 5.028e-06 |
| Max relative difference | 1.000e+00 |
| Octave/reference blow-up values | 1 |
| Blow-up values where scipy_chunked remains bounded | 1 |

---

## 8. Top Differences

| Rank | Longitude | Latitude | Date | Octave/reference | scipy_chunked | Relative difference |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 10 | 46 | 2024-12-23 | 6.92034e+15 | 0 | 1.000e+00 |
| 2 | 10 | 46 | 2024-12-21 | 341640 | 0 | 1.000e+00 |
| 3 | 10 | 46 | 2024-12-22 | 85.0641 | 4.08365e-24 | 9.884e-01 |
| 4 | 9 | 45.5 | 2024-11-21 | 53.8319 | 53.7962 | 6.509e-04 |
| 5 | 8.5 | 45 | 2024-11-21 | 1.90514 | 1.90377 | 4.698e-04 |

---

## 9. Recommendation

- Use `legacy_optimized` when the goal is Octave-compatible production behavior.
- Use `scipy_chunked` as the preferred adaptive-solver candidate for modernization.
- Do not judge SciPy only by whether it reproduces the Octave blow-up.
- Define scientific acceptance criteria for bounded behavior in unstable regions before replacing the compatibility backend.
