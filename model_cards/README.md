
---
language: en
license: mit
tags:
- epidemiology
- vector-borne-disease
- climate-health
- mosquito-modeling
library_name: custom
pipeline_tag: time-series-forecasting
---

# HeiPlanet Vector & Disease Modeling Suite

This repository contains epidemiological and ecological models developed
by **HeiPlanet** to support climate-sensitive vector-borne disease risk
assessment.

## Included Models

### WNV-R0
**Task:** Basic Reproduction Number (R₀) Estimation  
**Domain:** West Nile Virus  
**Description:** Estimates seasonal transmission potential using climate,
entomological, and host population data.

### Aedes albopictus Population Model
**Task:** Vector Population Dynamics Estimation  
**Domain:** Vector Ecology  
**Description:** Estimates mosquito population abundance across life stages
using temperature-dependent development and environmental drivers.

## Intended Use
- Public health preparedness and surveillance planning
- Climate and environmental scenario analysis
- Research and decision support (not real-time operations)

## Limitations
- Seasonal (non–real-time) models
- Performance depends on data availability and quality
- Outputs require expert interpretation

## Ethical & Safety Considerations
- Do not use as the sole basis for public health decisions
- Consider surveillance and socioeconomic disparities
- Communicate uncertainty in all downstream use

## Maintenance
Models are updated annually or seasonally as new data become available.
Older versions are maintained for two years.

## Contact
HeiPlanet – your-email@organization.com
