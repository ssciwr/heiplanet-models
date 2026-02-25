
# Model Card: WNV-R0

## Overview
**Model Name:** WNV-R0  
**Version:** 1.0  
**Type:** Process-based epidemiological model
**Task:** Thermal suitability Estimation  
**Domain:** West Nile Virus (WNV) Transmission Potential

**Organization:** HeiPlanet  
**Date Created:** 2026-01-12  
**Contact:** julian.heidecke@iwr.uni-heidelberg.de

## Description
WNV-R0 estimates a relative version of the basic reproduction number (R₀) 
for WNV transmission by Culex pipiens using ambient temperature data. 
The model is intended for seasonal and regional temperature suitability 
assessments rather than real-time forecasting of WNV outbreaks.

## Intended Use
### In Scope
- Temperature-driven seasonal WNV risk assessment
- Regional comparison of thermal suitability
- Historical warming impact assessments on transmission potential
- Climate scenario analysis
- Public health preparedness planning

### Out of Scope
- Individual risk prediction
- Real-time outbreak forecasting
- Automated policy decisions

## Architecture
- Ross-Macdonald type relative reproduction number
- Derived from deterministic compartmental vector–host model  
- Temperature-dependent mosquito-pathogen parameters for Culex pipiens

## Inputs
| Feature | Description | Units |
|-------|------------|-------|
| Temperature | Monthly mean temperature | °C |

## Output
- **Relative R₀ Estimate:** Dimensionless relative reproduction number

## Training Data
- Parameterization of the model is based on laboratory experimental data on the 
temperature dependence of mosquito-pathogen traits
- Temperature response curves were fitted using Bayesian hierarchical models

## Validation
**Data:** Historical WNV outbreak records in Europe
**Metrics:** Seasonal and geographical alignment between R0 and cases, 
rank correlation coefficients, and overlap between lab-based and field-observed 
"optimal" temperature for transmission

## Assumptions
- R₀ is a measure of long-term average transmission under constant temperatures
- Homogeneous spatial mixing
- No explicit modeling of interventions
- No intra-species mosquito variability in temperature sensitivity

## Limitations
- Deterministic structure
- Neglects host community composition and immunity dynamics
- Only focused on temperature-driven effects on transmission potential via mosquito-pathogen traits
- Accurate risk predictions need to account for additional climatic factors
- Cannot predict number of cases 
- Cannot account for potential adaptation of mosquito populations to increasing temperatures
- Relative R₀ cannot be interpreted as a threshold parameter like absolute R₀

## Bias & Fairness
- Based on laboratory data
- Only validated against WNV observations in Europe (and to some extent in the USA)

## Potential Harms
- Misuse without expert review
- Misinterpretation of outputs

## Risk Mitigation
- Expert review
- Communication of limitations and uncertainty

## Maintenance
- Updates as new data become available
- Continued methodological updates and model extensions

## Datasets
- Laboratory experimental data on mosquito-pathogen traits compiled through 
systematic literature review
- Copernicus ERA5-Land Climate Data
- ECDC human West Nile Neuroinvasive disease cases data for validation

## Publications & Related Models
- Heidecke, J., Wallin, J., Fransson, P., Singh, P., Sjödin, H., Stiles, P. C., ... & Rocklöv, J. (2025). 
Uncovering temperature sensitivity of West Nile virus transmission: 
Novel computational approaches to mosquito-pathogen trait responses. 
PLOS Computational Biology, 21(3), e1012866.
- Heidecke, J., Fransson, P., Wallin, J., & Rocklöv, J. Thermal Biology-Informed 
Reproduction Number Explains Spatiotemporal Patterns of West Nile Incidence in Europe. 
Available at SSRN 5597581.
