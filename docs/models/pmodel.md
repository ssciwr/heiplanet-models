# Pmodel (Aedes albopictus)

## Overview

Pmodel is a climate-driven, spatial population model for *Aedes albopictus*. It combines daily temperature, rainfall, and population-density data to solve five mosquito population compartments over a configured year range. Results are written as a NetCDF dataset with one variable per compartment.

- **Production runner**: [`scripts/run_pmodel_pipeline.py`](../../scripts/run_pmodel_pipeline.py).
- **Configuration file**: [`src/heiplanet_models/Pmodel/global_settings.yaml`](../../src/heiplanet_models/Pmodel/global_settings.yaml).

Key features:

!!! note

    `scipy_chunked` is the current **production** solver backend.
    `legacy` and `legacy_optimized` solver backends are used to compare numerically existing Python migration to the original code written in Octave.

- Supports `scipy_chunked`, `legacy`, and `legacy_optimized` solver backends.
- Uses spatial chunking for the default `scipy_chunked` backend.
- Emits warnings for skipped carrying-capacity cells and numerical recovery.
- Creates default initial conditions when no usable initial-conditions file is configured.

**Reference Publication:**

- DOI: [https://doi.org/10.1038/s43247-025-02199-z](https://doi.org/10.1038/s43247-025-02199-z)
- [Supplementary material](https://static-content.springer.com/esm/art%3A10.1038%2Fs43247-025-02199-z/MediaObjects/43247_2025_2199_MOESM2_ESM.pdf)

---

## Run the Model

1. From the repository root, synchronize the development environment:

```bash
uv sync
```

2. Configure the following fields in [`global_settings.yaml`](../../src/heiplanet_models/Pmodel/global_settings.yaml):

    ```yaml
    ingestion:
        path_root_datasets: "data/in/Pratik_datalake"

    serving:
        path_output_datasets: "data/out/Pratik_datalake"

    execution:
        initial_year: 2024
        final_year: 2024
    ```

    `initial_year` and `final_year` are inclusive. The Pmodel runner intentionally
    does not provide command-line year overrides; use a separate settings file for
    an ad hoc year range.

3. Run the default production backend:

    ```bash
    uv run python scripts/run_pmodel_pipeline.py
    ```

Use another settings file when needed:

```bash
uv run python scripts/run_pmodel_pipeline.py --settings path/to/settings.yaml
```

Display available options:

```bash
uv run python scripts/run_pmodel_pipeline.py --help
```

### Solver Backends

| Backend | Use case | Notes |
| --- | --- |--- |
| `scipy_chunked` | Production backend (faster) |Uses `scipy.integrate.solve_ivp`<br>and spatial chunks. | |
| `legacy` | Reproduce legacy behavior |Original RK4-style policy. |
| `legacy_optimized` | Legacy comparison with precomputed rates | Numerically aligned with `legacy`. |

Example SciPy run with smaller spatial chunks:

```bash
uv run python scripts/run_pmodel_pipeline.py \
    --backend scipy_chunked \
    --chunk-lon 72 \
    --chunk-lat 72 \
    --scipy-method RK45 \
    --scipy-rtol 1e-5 \
    --scipy-atol 1e-5
```

`--chunk-lon`, `--chunk-lat`, `--scipy-method`, `--scipy-rtol`, and
`--scipy-atol` apply only to `scipy_chunked`. Supplying non-default values with
a legacy backend logs a warning and leaves the legacy solver behavior unchanged.

---

## Benchmark the Pipeline

[`scripts/benchmark_pmodel_pipeline.py`](../../scripts/benchmark_pmodel_pipeline.py)
is a developer tool for measuring complete Pmodel runs. It measures each pipeline
stage and reports wall time, CPU time, peak resident-memory delta, and process
read/write I/O. It also writes the normal NetCDF output configured in `serving`.

Unlike the production runner, the benchmark selects years with command-line
arguments. Its `--initial-year` and `--final-year` values take precedence over
the `execution` section in the settings file.

Run a one-year benchmark using the default `legacy_optimized` backend:

```bash
uv run python scripts/benchmark_pmodel_pipeline.py \
    --settings src/heiplanet_models/Pmodel/global_settings.yaml \
    --initial-year 2024 \
    --final-year 2024
```

Benchmark the production backend with smaller chunks while investigating
performance:

```bash
uv run python scripts/benchmark_pmodel_pipeline.py \
    --settings src/heiplanet_models/Pmodel/global_settings.yaml \
    --initial-year 2024 \
    --final-year 2024 \
    --backend scipy_chunked \
    --chunk-lon 72 \
    --chunk-lat 72
```

Useful options:

| Option | Purpose |
| --- | --- |
| `--backend` | Select `legacy`, `legacy_optimized`, or `scipy_chunked`. The benchmark defaults to `legacy_optimized`. |
| `--initial-year`, `--final-year` | Inclusive range to benchmark. |
| `--chunk-lon`, `--chunk-lat` | Spatial chunk dimensions for `scipy_chunked`. |
| `--scipy-method`, `--scipy-rtol`, `--scipy-atol` | SciPy solver configuration for `scipy_chunked`. |
| `--fail-fast` | Stop after the first failed year. |
| `--verbose` | Show a traceback for failed years. |
| `--progress` / `--no-progress` | Enable or disable live stage messages. |
| `--chunk-progress` / `--no-chunk-progress` | Enable or disable per-chunk progress. It is enabled by default for `scipy_chunked`. |
| `--quiet-progress` | Disable all live stage and chunk messages. |

For a clean summary suitable for comparing repeated runs:

```bash
uv run python scripts/benchmark_pmodel_pipeline.py \
    --initial-year 2024 \
    --final-year 2024 \
    --backend scipy_chunked \
    --quiet-progress
```

The final report contains:

- `Stage Measurements`: one row per stage and year.
- `Bottleneck Summary`: stage metrics aggregated across all requested years,
  sorted by wall time.
- `Year Summary` and `Benchmark Totals`: completed, skipped, failed, and total
  duration information.

Use the benchmark with representative production input and an output directory
that can be overwritten. It runs the model and is not a dry run.

---

## Configuration and Data Contract

The runner creates input filenames as:

```text
<prefix><year><suffix><extension>
```

For the default temperature settings and year `2024`, this is:

```text
data/in/Pratik_datalake/ERA5land_global_t2m_daily_0.5_2024.nc
```

`ode_system.time_step` must be a positive integer. It means sub-steps per day
for legacy backends; the SciPy backend maintains the same public setting for
API consistency. `model_variables` must remain in this order:

```yaml
model_variables:
    - eggs
    - ed
    - juv
    - imm
    - adults
```

Set `ingestion.initial_conditions.file_path_initial_conditions` to a NetCDF
file to start from its final time slice. If it is empty or does not point to an
existing file, Pmodel generates its default initial-condition array.

### Inputs

| Dataset                  | Description              | Format  |
| ------------------------ | ------------------------ | ------- |
| Temperature | Daily gridded air temperature | NetCDF |
| Rainfall | Daily gridded precipitation | NetCDF |
| Human population | Gridded population density | NetCDF |
| Initial conditions (optional) | Starting state; final time slice is used | NetCDF |

### Outputs
The output file is written to `serving.path_output_datasets` with the configured
prefix, year, suffix, and extension. It contains these variables, each with
`(longitude, latitude, time)` dimensions:

| Variable | Description |
| --- | --- |
| `eggs` | Non-diapause eggs |
| `ed` | Diapause eggs |
| `juv` | Juveniles |
| `imm` | Immature adults |
| `adults` | Mature adults |

### Numerical Warnings

The runner reports cells skipped by `scipy_chunked` when carrying capacity is
non-positive or non-finite; skipped cells are written as zero. It also reports
any non-finite derivative substitutions or negative/non-finite output values
normalized to zero. These warnings preserve the existing model behavior and
make numerical recovery auditable.

### Tests

Run Pmodel unit and integration tests from the repository root:

```bash
uv run pytest test/unit/Pmodel test/integration/Pmodel -q
```

## Mathematical Background

### Description

The source implementation currently stores five compartments: non-diapause eggs,
diapause eggs, juveniles, immature adults, and mature adults. The detailed
equations below are retained as background material from earlier model
documentation and should be reconciled with the current five-compartment
implementation before being used as an exact implementation specification.

Historical six-stage differential equation model:

- Three aquatic stages:
    - Egg $E$
    - Diapaussing egg $E_{\text{dia}}$
    - Juvenile stage (Larval stage + Pupal stage)

- Three aerial stages:
    - Emerging adult $A_{\text{em}}$
    - Blood fed adults $A_{b}$
    - Ovipositing adults $A_{0}$

### System of Equations

| Differential Equation                                                                                                                                                      |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $\dot{E} = \beta(T)(1-\omega(\overline{T}, \overline{S}))M - Q(W,P)\delta_{E}(T)E - m_{E}(T)E$                                                                             |
| $\dot{E_{\text{dia}}} = \delta_{E}(T)\omega(\overline{T}, \overline{S})E_{\text{dia}} - \sigma(\overline{T},S)Q(W,P)\delta_{E}(T)E_{\text{dia}} - m_{Ed}(T)E_{\text{dia}}$ |
| $\dot{J} = Q(W,P)\delta_{E}(T)E + \sigma(\overline{T},S)Q(W,P)\delta_{E}(T)E_{\text{dia}} - \delta_{J}(T)J - \left( \frac{J}{K_{J}(W,P)} + m_{J}(T) \right)J$              |
| $\dot{A_{\text{em}}} = \frac{1}{2}\delta_{J}(T)J - \delta_{A_{\text{em}}}(T)A_{\text{em}} - m_{A}(T)A_{\text{em}}$                                                         |
| $\dot{A_{b}} = \delta_{A_{\text{em}}}(T)A_{\text{em}} + \delta_{A_{o}}A_{o} - \left( m_{A}(T) + r + \delta_{A_{b}}(T) \right)A_{b}$                                        |
| $\dot{A_{o}} = \delta_{A_{b}}(T)A_{b} - \left( m_{A}(T) + r + \delta_{A_{o}}A_{o} \right)$                                                                                 |

### Variables and Parameters

#### State Variables

| Variable         | Description                          | Units       |
| ---------------- | ------------------------------------ | ----------- |
| $E$              | Egg population                       | individuals |
| $E_{\text{dia}}$ | Diapaussing egg population           | individuals |
| $J$              | Juvenile population (larvae + pupae) | individuals |
| $A_{\text{em}}$  | Emerging adult population            | individuals |
| $A_{b}$          | Blood fed adult population           | individuals |
| $A_{o}$          | Ovipositing adult population         | individuals |

#### Environmental Inputs

| Variable       | Description                                            | Units      |
| -------------- | ------------------------------------------------------ | ---------- |
| $T$            | Temperature                                            | °C         |
| $S$            | Photoperiod (day-light length). Using *Forsythe model* | hours      |
| $W$            | Precipitation/Rainfall                                 | mm         |
| $P$            | Human population density                               | people/km² |
| $\overline{T}$ | Mean temperature over previous 7 days                  | °C         |
| $\overline{S}$ | Mean day-light over previous 7 days                    | hours      |


### Equations in detail

| Number | Function                             | Description                                                              | Units         |
| :----: | ------------------------------------ | ------------------------------------------------------------------------ | ------------- |
|   1    | $\beta(T)$                           | Egg per female per day                                                   | $\frac{1}{day}$         |
|   2    | $\omega(\overline{T}, \overline{S})$ | Diapausing egg proportion                                                | `NA`          |
|   3    | $Q(W,P)$                             | Hatching fraction depending in human density and rainfall                | `NA`          |
|   4    | $\delta_{E}(T)$                      | Egg development rate                                                     | $\frac{1}{day}$         |
|   5    | $m_{E}(T)$                           | Egg mortality rate                                                       | $\frac{1}{day}$         |
|   6    | $\sigma(\overline{T}, S)$            | Spring hatching rate                                                     | $\frac{1}{day}$         |
|   7    | $m_{Ed}(T)$                          | Diapausing egg mortality rate                                            | $\frac{1}{day}$         |
|   8    | $\delta_{J}(T)$                      | Juvenile development rate                                                | $\frac{1}{day}$         |
|   9    | $K_{J}(W,P)$                         | Juvenile Carrying Capacity                                               | `NA`          |
|   10   | $m_{J}(T)$                           | Juvenile Mortality rate                                                  | $\frac{1}{day}$         |
|   11   | $\delta_{A_{em}}(T)$                 | Emerging adult development rate                                          | $\frac{1}{day}$         |
|   12   | $m_{A}(T)$                           | Adult Mortality rate                                                     | $\frac{1}{day}$         |
|   13   | $\delta_{A_{o}}$                     | Subsequent blood meal rate after oviposition                             | $\frac{1}{day}$         |
|   14   | $r$                                  | Mortality rate associated with long distance travel  and search behavior | $\frac{1}{day}$         |
|   15   | $\delta_{A_{b}}(T)$                  | Blood fed adult development rate                                         | $\frac{1}{day}$         |
|   16   | $f(X)$                               | Sigmoidal "step-function"                                                | dimensionless |
|   17   | $\check{S}_{a}$                      | Critical day-light length in autumn                                      | hours         |
|   18   | $\check{T}_{D}$                      | Critical diapause temperature (°C)                                       | °C            |
|   x    | $M$                                  | `unknown`                                                                | `unknown`     | 


#### **1. Egg per female per day**

$$
\begin{align}
\beta(T) &= \max(-0.0163 + 1.2897T -15.837T^{2}, 0)
\end{align}
$$

| Constant | Description | Typical Value |
| -------- | ----------- | ------------- |
| $k1$     | *---*       | -0.0163       |
| $k2$     | *---*       | 1.289         |
| $k3$     | *---*       | -15.837       |

---

#### **2. Diapausing egg proportion**

$$
\begin{align}
\omega(\underline{T}, \underline{S}) = 0.5 \times f\left(\underline{S} - \check{S}_{a}\right)\, f\left(-\underline{T} - \check{T}_{D}\right)
\end{align}
$$

| Constant | Description | Typical Value |
| -------- | ----------- | ------------- |
| $k1$     | *---*       | 0.5           |

---

#### **3. Hatching fraction depending in human density and rainfall**

$$
\begin{align}
Q(W, P) = 0.8 \left( \frac{2.5\, e^{-0.05\,(W(t)-8)^2}}{e^{-0.05\,(W(t)-8)^2} + 1.5} \right) + 0.2 \left( \frac{0.01}{0.01 + e^{-0.01 P(t)}} \right)
\end{align}
$$

| Constant                                                 | Description | Typical Value |
| -------------------------------------------------------- | ----------- | ------------- |
| $E_{\text{opt}}$ *taken from octave code water_hatch.m*  | *---*       | 8             |
| $E_{\text{var}}$ *taken from octave code water_hatch.m*  | *---*       | 0.05          |
| $E_{\text{0}}$ *taken from octave code water_hatch.m*    | *---*       | 1.5           |
| $E_{\text{rat}}$ *taken from octave code water_hatch.m*  | *---*       | 0.02          |
| $E_{\text{dens}}$ *taken from octave code water_hatch.m* | *---*       | 0.01          |
| $E_{\text{fac}}$ *taken from octave code water_hatch.m*  | *---*       | 0.01          |

---

#### **4. Egg development rate**

$$
\begin{align}
\delta_{E}(T) = 0.5070\left( - \left( \frac{T-30.85}{12.82} \right)^2 \right)
\end{align}
$$

| Constant | Description | Typical Value |
| -------- | ----------- | ------------- |
| $k1$     | *---*       | 0.5070        |
| $k2$     | *---*       | 30.85         |
| $k3$     | *---*       | 12.82         |

---

#### **5. Egg mortality rate**

$$
\begin{align}
m_{E}(T) = -\ln\left( 0.955\, \exp\left[ -0.5 \left( \frac{T - 18.8}{21.53} \right)^{6} \right] \right)
\end{align}
$$

| Constant | Description | Typical Value |
| -------- | ----------- | ------------- |
| $k1$     | *---*       | 0.955         |
| $k2$     | *---*       | 0.5           |
| $k3$     | *---*       | 18.8          |
| $k4$     | *---*       | 21.53         |

---

#### **6. Spring hatching rate**

$$
\begin{align}
\sigma(\overline{T}, S) = 0.1 \times f(\check{T} - \underline{T})\, f(-\check{S}_{s} - S)
\end{align}
$$

| Constant | Description | Typical Value |
| -------- | ----------- | ------------- |
| $k1$     | *---*       | 0.1           |

---

#### **7. Diapausing egg mortality rate**

$$
\begin{align}
m_{Ed}(T) = m_{E}(T) = -\ln\left( 0.955\, \exp\left[ -0.5 \left( \frac{T - 18.8}{21.53} \right)^{6} \right] \right)
\end{align}
$$

| Constant | Description | Typical Value |
| -------- | ----------- | ------------- |
| $k1$     | *---*       | 0.955         |
| $k1$     | *---*       | 0.5           |
| $k3$     | *---*       | 18.8          |
| $k4$     | *---*       | 21.53         |

---

#### **8. Juvenile development rate**

$$
\begin{align}
\delta_{J}(T) = \frac{1}{0.08T^{2} - 4.89T + 83.85}
\end{align}
$$

| Constant | Description | Typical Value |
| -------- | ----------- | ------------- |
| $k1$     | *---*       | 0.08          |
| $k2$     | *---*       | -4.89         |
| $k3$     | *---*       | 83.85         |

---

#### **9. Juvenile carrying capacity**

$$
\begin{align}
K_{J}(W,P) = \lambda\,\frac{0.1}{1 - 0.9^{t}}\sum_{x=1}^{t} 0.9^{(t-x)}\left(\alpha_{\text{rain}}W(x) + \alpha_{\text{dens}}P(x)\right)
\end{align}
$$

| Constant                                     | Description                                | Typical Value |
| -------------------------------------------- | ------------------------------------------ | ------------- |
| $\lambda$                                    | Scaling factor of carrying capacity        | $10^6$        |
| $t$                                          | ---                                        | -             |
| $\alpha_{\text{rain}}$                       | Weight for rainfall contribution           | $10^{-3}$     |
| $W(x)$                                       | Rainfall at time step $x$                  | -             |
| $\alpha_{\text{dens}}$                       | Weight for population contribution         | $10^{-5}$     |
| $P(x)$                                       | Humand density population at time step $x$ | -             |
| $\gamma$ *taken from octave code capacity.m* | ---                                        | 0.9           |

---

#### **10. Juvenile mortality rate**

$$
\begin{align}
m_{J}(T) = -\ln\left[\,0.977\, \exp\left(-0.5 \left(\frac{T - 21.8}{16.6}\right)^{6}\right)\,\right]
\end{align}
$$

| Constant | Description | Typical Value |
| -------- | ----------- | ------------- |
| $k1$     | *---*       | 0.977         |
| $k2$     | *---*       | 0.5           |
| $k3$     | *---*       | 21.8          |
| $k4$     | *---*       | 16.6          |

---

#### **11. Emerging adult development rate**

$$
\begin{align}
\delta_{A_{em}}(T) = \frac{1}{0.069T^2 - 3.574T + 50.1}
\end{align}
$$

| Constant | Description | Typical Value |
| -------- | ----------- | ------------- |
| $k1$     | *---*       | 0.069         |
| $k2$     | *---*       | 3.574         |
| $k3$     | *---*       | 50.1          |

---

#### **12. Adult mortality rate**

$$
\begin{align}
m_{A}(T_{\text{mean}}) = -\ln\left[\,0.677\, \exp\left(-0.5 \left(\frac{T_{\text{mean}} - 20.9}{13.2}\right)^{6}\right)\, (T_{\text{mean}})^{0.1}\,\right]
\end{align}
$$

| Constant | Description | Typical Value |
| -------- | ----------- | ------------- |
| $k1$     | *---*       | 0.677         |
| $k2$     | *---*       | 0.5           |
| $k3$     | *---*       | 20.9          |
| $k4$     | *---*       | 13.2          |
| $k5$     | *---*       | 0.1           |

---

#### **13. Subsequent blood meal rate after oviposition**

$$
\begin{align}
\delta_{A_{o}} = k_{1}
\end{align}
$$

| Constant | Description | Typical Value |
| -------- | ----------- | ------------- |
| $k_{1}$  | *---*       | 0.1           |

---

**14. Mortality rate associated with long distance travel and search behavior**

$$
\begin{align}
r = k_{1}
\end{align}
$$

| Constant | Description | Typical Value |
| -------- | ----------- | ------------- |
| $k_{1}$  | *---*       | 0.8           |

---

**15. Blood fed adult development rate**

$$
\begin{align}
\delta_{Ab} = \frac{T-10}{77}+0.2
\end{align}
$$

| Constant | Description | Typical Value |
| -------- | ----------- | ------------- |
| $k_{1}$  | *---*       | 10            |
| $k_{2}$  | *---*       | 77            |
| $k_{3}$  | *---*       | 0.2           |

---

#### **16. Sigmoidal "step function"**

$$
\begin{align}
f(X)=\frac{1}{1+e^{20X}}
\end{align}
$$

| Constant | Description | Typical Value |
| -------- | ----------- | ------------- |
| $k_{1}$  | *---*       | 20            |


#### **17. Critical day-light length in autumn**

$$
\begin{aligned}
\check{S}_{a}=10.058+0.08965 \times Latitude(degrees)
\end{aligned}
$$

| Constant | Description | Typical Value |
| -------- | ----------- | ------------- |
| $k_{1}$  | *---*       | 10.058        |
| $k_{2}$  | *---*       | 0.08965       |

---

#### **18. Critical diapause temperature (°C)**

$$
\begin{align}
\check{T}_{D} = k_{1}
\end{align}
$$

| Constant | Description | Typical Value |
| -------- | ----------- | ------------- |
| $k_{1}$  | *---*       | 21            |

---

## Authors & Contact

Here is a markdown table template for author information:

| Author      | GitHub Username   | Email                                                                  | Affiliation                                 |
| ----------- | ----------------- | ---------------------------------------------------------------------- | ------------------------------------------- |
| Pratik Singh |  | [pratik.singh@uni-heidelberg.de](mailito:pratik.singh@uni-heidelberg.de) | [Hei-Planet Planetary Health Hub](https://hei-planet.com/) |


| Collaborator      | GitHub Username   | Email                                                                  | Affiliation                                 |
| ----------- | ----------------- | ---------------------------------------------------------------------- | ------------------------------------------- |
| Edwin Carreño | @ecarrenolozano | [edwin.carreno@iwr.uni-heidelberg.de](mailito:edwin.carreno@iwr.uni-heidelberg.de) | [Scientific Software Center](https://www.ssc.uni-heidelberg.de/en) |

