# Model Card: Aedes albopictus Population Model

## Overview
**Model Name:** Aedes albopictus Population Model (Pmodel)  
**Version:** 1.0  
**Type:** Process-based entomological model (stage-structured, climate-driven)  
**Task:** Vector Abundance / Climatic Suitability Estimation  
**Domain:** Aedes albopictus Population Dynamics  

**Organization:** HeiPlanet  
**Date Created:** 2026-01-12  
**Contact:** ssc@uni-heidelberg.de  

## Description
The Aedes albopictus Population Model (Pmodel) is a mechanistic, stage-structured
life cycle model that simulates daily Aedes albopictus abundance per grid cell from
temperature, rainfall and human population density. It is the mechanistic life
cycle model used as a covariate in the spatio-temporal diffusion model of
Barman et al. (2025), which predicts new establishments of Ae. albopictus in Europe.
The model estimates climate-driven vector suitability and abundance; on its own it
does not model introduction or spread of the mosquito into new areas.

## Intended Use
### In Scope
- Climate-driven suitability and relative abundance mapping of Ae. albopictus
- Mechanistic covariate for statistical establishment and early-warning models
- Comparison of seasonal and inter-annual abundance patterns across regions
- Climate scenario and climate impact assessments

### Out of Scope
- Stand-alone forecasting of introduction or spread into uncolonised areas
- Absolute, trap-level mosquito count prediction
- Real-time operational vector control
- Automated policy decisions

## Architecture
- Deterministic system of ordinary differential equations, solved independently per grid cell
- Five compartments: non-diapause eggs, diapause eggs, juveniles, immature adults, mature adults
- Temperature-dependent birth, development and mortality rates
- Diapause induction and spring hatching driven by photoperiod and 7-day mean temperature
- Photoperiod computed from latitude and day of year (Forsythe model)
- Egg hatching and juvenile carrying capacity driven by rainfall and human population density
- Based on the Metelmann et al. (2019) / Erguler et al. (2016) model family
- Production solver `scipy.integrate.solve_ivp` with spatial chunking; legacy RK4 backends reproduce the original Octave code

## Inputs
| Feature | Description | Units |
|-------|------------|-------|
| Temperature | Daily mean 2 m air temperature (ERA5-Land) | °C |
| Precipitation | Daily total precipitation (ERA5-Land) | mm |
| Human Population Density | Gridded human population density | people/km² |
| Initial Conditions (optional) | Starting state per compartment; final time slice is used, otherwise defaults are generated | mosquitoes/grid cell |

## Outputs
Daily values per grid cell (NetCDF, dimensions longitude × latitude × time):

| Variable | Description | Units |
|-------|------------|-------|
| `adults` | Mature adults | mosquitoes/grid cell |
| `imm` | Immature (emerging) adults | mosquitoes/grid cell |
| `juv` | Juveniles (larvae and pupae) | mosquitoes/grid cell |
| `eggs` | Non-diapause eggs | mosquitoes/grid cell |
| `ed` | Diapause eggs | mosquitoes/grid cell |

## Training Data
- No model fitting; rate functions taken from published laboratory and field studies on Ae. albopictus life-history traits
- Parameterisation follows Metelmann et al. (2019), Erguler et al. (2016), Jia et al. (2016), Tran et al. (2013), Cailly et al. (2012)

## Evaluation
**Testing Data:** Indirect validation only. Yearly adult abundance aggregated to
NUTS3/GAUL regions was used as a covariate in a Bayesian GAM (R-INLA) predicting new
Ae. albopictus establishments, evaluated against ECDC/EFSA VectorNet presence/absence
records 2010–2023.

**Metrics:**
- Rolling-window (one year ahead, 2017–2023) AUC for new establishments: 0.79
- Fixed-window AUC: 0.73 (test 2013–2017), 0.84 (test 2019–2023)
- Performance comparable to raw climate and population covariates (rolling-window AUC 0.80)
- No validation against field abundance counts

## Assumptions
- Well-mixed mosquito population within each grid cell
- No dispersal between grid cells
- Fixed, literature-derived parameters without local adaptation
- Human population density as a proxy for breeding-site availability
- No explicit modeling of vector control interventions

## Limitations
- Estimates suitability and abundance, not presence; cannot tell whether the species has been introduced
- Not validated against field abundance data
- Humidity not included, although relative humidity was a relevant predictor in Barman et al. (2025)
- Results depend on initial conditions
- Grid cells with non-positive or non-finite carrying capacity are skipped and written as zero
- Repository default configuration (0.5° grid, 10 sub-steps per day, daily mean forcing) differs from the published setup (0.25° grid, 100 sub-steps per day, hourly temperature in R deSolve)

## Bias & Fairness
- Validation data unevenly distributed; many "no data" regions on the Iberian Peninsula (south-west AUC 0.60–0.69)
- Surveillance effort differs between countries
- Only validated in Europe

## Potential Harms
- Misreading abundance as confirmed presence
- Misallocation of vector control resources
- Underestimation of risk in data-poor regions
- Misuse without expert review

## Risk Mitigation
- Expert review by vector control specialists
- Combine with surveillance data and establishment models
- Communication of limitations and uncertainty

## Maintenance
- Updates as new data become available
- Continued methodological updates and model extensions

## Datasets
- Copernicus ERA5-Land climate data
- GPWv4 gridded human population density
- ECDC/EFSA VectorNet Ae. albopictus presence/absence data (validation only, available on request from ECDC)

## Publications & Related Models
**Related Models:** WNV-R0  
**Publications:**
- Barman, S., Semenza, J. C., Singh, P., et al. (2025). A climate and population dependent diffusion model forecasts the spread of Aedes albopictus mosquitoes in Europe. *Communications Earth & Environment*, 6, 276. https://doi.org/10.1038/s43247-025-02199-z
- Metelmann, S., Caminade, C., Jones, A. E., Medlock, J. M., Baylis, M., & Morse, A. P. (2019). The UK's suitability for Aedes albopictus in current and future climates. *Journal of the Royal Society Interface*, 16, 20180761.
