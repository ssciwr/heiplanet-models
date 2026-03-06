
# Model Card: Aedes albopictus Population Model

## Overview
**Model Name:** Aedes albopictus Population Model  
**Version:** 1.0  
**Type:** Ecological / Entomological (Stage-Structured Population Model)  
**Task:** Vector Population Dynamics Estimation  
**Domain:** Aedes albopictus Ecology and Surveillance  

**Organization:** HeiPlanet  
**Date Created:** 2026-01-12  
**Contact:** your-email@organization.com  

## Description
This model estimates Aedes albopictus population abundance across life stages
using temperature-dependent development rates, survival probabilities, and
environmental drivers. It is designed for seasonal forecasting and risk assessment.

## Intended Use
### In Scope
- Ecological forecasting of vector abundance
- Vector surveillance and control planning
- Disease transmission risk assessment
- Climate and habitat scenario analysis

### Out of Scope
- Real-time operational vector control
- Individual mosquito tracking

## Architecture
- Deterministic, stage-structured population model
- Temperature-dependent development and survival
- Environmental forcing (temperature, precipitation, humidity)
- Photoperiod-driven seasonality

## Inputs
| Feature | Description | Units |
|-------|------------|-------|
| Temperature | Daily mean temperature | °C |
| Precipitation | Daily precipitation | mm |
| Humidity | Relative humidity | % |
| Habitat Area | Breeding habitat availability | m² |
| Larval Density | Larval population density | larvae/habitat |
| Photoperiod | Day length | hours |

## Outputs
- Egg density (eggs/habitat)
- Larval density (larvae/habitat)
- Pupal density (pupae/habitat)
- Adult density (adults/100 m³)
- Finite population growth rate

## Training Data
- Field entomological surveys
- ERA5 climate reanalysis
- Laboratory rearing experiments
- Published thermal biology data  

_Preprocessing includes climate-based imputation, rate calculations,
photoperiod normalization, and habitat suitability classification._

## Evaluation
**Metrics:** RMSE, Mean Absolute Percentage Error, Temporal Correlation (TBD)  
**Testing Data:** Field-collected population observations

## Assumptions
- Deterministic population dynamics
- Seasonal environmental stability
- Simplified density dependence
- No insecticide resistance modeling

## Limitations
- Uneven geographic surveillance coverage
- Simplified habitat and biological processes
- Reduced performance in arid and highly urbanized areas
- Seasonal (non–real-time) design

## Bias & Fairness
- Biased toward regions with stronger entomological monitoring
- Surveillance disparities may affect estimated risk

## Potential Harms
- Misallocation of vector control resources
- Overconfidence without field validation
- Underestimation of risk in data-poor regions

## Risk Mitigation
- Expert review by vector control specialists
- Uncertainty quantification
- Regular field validation and recalibration

## Maintenance
- Seasonal updates or as new data become available
- Semantic versioning
- Two-year deprecation window

## Datasets
- Field entomological survey data
- Copernicus ERA5 Climate Data
- Published Aedes albopictus thermal biology studies

## Publications & Related Models
**Related Models:** WNV-R0, Dengue transmission models  
**Publications:** None currently listed
