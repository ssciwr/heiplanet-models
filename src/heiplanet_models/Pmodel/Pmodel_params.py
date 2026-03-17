# -----------------------------------------------
# ----      General function constants       ----
# -----------------------------------------------
MIN_LAT_DEGREES = -90
MAX_LAT_DEGREES = 90
HOURS_PER_DAY = 24

DAYS_YEAR = 365
HALF_DAYS_YEAR = 183


# Ref: https://doi.org/10.1016/0304-3800(94)00034-F
CONSTANTS_REVOLUTION_ANGLE = {
    "CONST_1": 0.2163108,
    "CONST_2": 0.9671396,
    "CONST_3": 0.0086,
    "CONST_4": 367,
    "CONST_5": 186,
}

# Ref: https://doi.org/10.1016/0304-3800(94)00034-F
CONSTANT_DECLINATION_ANGLE = 0.39795

CONSTANTS_INITIAL_CONDITIONS = {
    "CONST_K1": 625,
    "CONST_K2": 100,
}

# ----------------------------------------------
# ----      Birth function constants        ----
# ----------------------------------------------
CONSTANTS_MOSQUITO_BIRTH = {
    "CONST_1": 38.8,
    "CONST_2": 33.2,
    "CONST_3": -0.5,
    "CONST_4": 70.3,
    "CONST_5": 14.1,
    "CONST_6": 1.5,
}

CONSTANTS_MOSQUITO_DIAPAUSE_LAY = {
    "RATIO_DIA_LAY": 0.5,
    "CONST_1": 10.058,
    "CONST_2": 0.08965,
    "DAYLENGTH_COEFFICIENT": 0,
}

CONSTANTS_MOSQUITO_DIAPAUSE_HATCHING = {
    "PERIOD": 7,
    "CPP": 11.25,
    "CTT": 11.0,
    "RATIO_DIA_HATCH": 0.1,
    "DAYLENGTH_COEFFICIENT": 0,
}

CONSTANTS_WATER_HATCHING = {
    "E_OPT": 8.0,
    "E_VAR": 0.05,
    "E_0": 1.5,
    "E_RAT": 0.2,
    "E_DENS": 0.01,
    "E_FAC": 0.01,
}

# ----------------------------------------------------
# ----      Development function constants        ----
# ----------------------------------------------------
CONSTANTS_MOSQUITO_J = {
    "CONST_1": 82.42,
    "CONST_2": 4.87,
    "CONST_3": 8e-2,
    "CONST_4": 1.0,
}

CONSTANTS_MOSQUITO_I = {
    "CONST_1": 50.1,
    "CONST_2": 3.574,
    "CONST_3": 0.069,
    "CONST_4": 1.0,
}

CONSTANTS_MOSQUITO_E = {
    "q": 0.0001246068,
    "T0": -7.0024634748,
    "Tm": 34.1519214674,
}

CONSTANTS_CARRYING_CAPACITY = {
    "ALPHA_RAIN": 1e-3,
    "ALPHA_DENS": 1e-5,
    "GAMMA": 9e-1,
    "LAMBDA": 1e6 * 625 * 100,
}

# ----------------------------------------------------
# ----       Mortality function constants         ----
# ----------------------------------------------------

CONSTANTS_MORTALITY_MOSQUITO_E = {
    "CONST_1": 0.955,
    "CONST_2": -0.5,
    "CONST_3": 18.8,
    "CONST_4": 21.53,
    "CONST_5": 6,
}

CONSTANTS_MORTALITY_MOSQUITO_J = {
    "CONST_1": 0.977,
    "CONST_2": -0.5,
    "CONST_3": 21.8,
    "CONST_4": 16.6,
    "CONST_5": 6,
}

CONSTANTS_MORTALITY_MOSQUITO_A = {
    "CONST_1": 0.677,
    "CONST_2": -0.5,
    "CONST_3": 20.9,
    "CONST_4": 13.2,
    "CONST_5": 6,
    "CONST_6": 0.1,
}


CONSTANTS_MORTALITY_MOSQUITO_ED = {
    "ED_SURV_BL": 1.0,
    "CONST_1": 0.93,
    "CONST_2": -0.5,
    "CONST_3": 11.68,
    "CONST_4": 15.67,
    "CONST_5": 6,
}
