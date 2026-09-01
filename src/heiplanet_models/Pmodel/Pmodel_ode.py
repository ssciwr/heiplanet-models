import logging
from collections.abc import Callable

import numpy as np
import xarray as xr

from heiplanet_models.Pmodel import Pmodel_initial
from heiplanet_models.Pmodel.Pmodel_rates_birth import (
    mosq_birth,
    mosq_dia_hatch,
    mosq_dia_lay,
)
from heiplanet_models.Pmodel.Pmodel_rates_development import (
    mosq_dev_i,
    mosq_dev_j,
)
from heiplanet_models.Pmodel.Pmodel_rates_mortality import (
    mosq_mort_a,
    mosq_mort_e,
    mosq_mort_j,
    mosq_surv_ed,
)

# ---- Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


MODEL_VARIABLES = 5  # eggs, diapause eggs, juveniles, immature adults, mature adults


def albopictus_ode_system(
    state: np.ndarray,
    model_params: tuple[
        int,  # time_index
        float,  # time_step
        np.ndarray,  # carrying_capacity
        np.ndarray,  # birth_rate
        np.ndarray,  # diapause_laying_fraction
        np.ndarray,  # diapause_hatching_fraction
        np.ndarray,  # egg_mortality
        np.ndarray,  # juvenile_mortality
        np.ndarray,  # adult_mortality
        np.ndarray,  # egg_diapause_survival
        np.ndarray,  # juvenile_development
        np.ndarray,  # immature_development
        float,  # egg_development
        np.ndarray,  # water_hatching_rate
    ],
) -> np.ndarray:
    """
    Computes the derivatives of the mosquito aedes albopictud population compartments for the ODE system.

    This function calculates the rates of change for each compartment (non-diapause eggs, diapause eggs,
    juveniles, immature adults, and mature adults) based on the current state and model parameters.

    Args:
        state (np.ndarray): Current state array of the system, representing the population in each compartment.
        model_params (tuple): Tuple containing model parameters and environmental variables in the following order:
            time_index (int): Current time index.
            time_step (float): Time step size.
            carrying_capacity (np.ndarray): Carrying capacity array.
            birth_rate (np.ndarray): Birth rate array.
            diapause_laying_fraction (np.ndarray): Fraction of eggs laid in diapause.
            diapause_hatching_fraction (np.ndarray): Fraction of diapause eggs hatching.
            egg_mortality (np.ndarray): Egg mortality rate array.
            juvenile_mortality (np.ndarray): Juvenile mortality rate array.
            adult_mortality (np.ndarray): Adult mortality rate array.
            egg_diapause_survival (np.ndarray): Survival rate of diapause eggs.
            juvenile_development (np.ndarray): Juvenile development rate array.
            immature_development (np.ndarray): Immature adult development rate array.
            egg_development (float): Egg development rate (scalar).
            water_hatching_rate (np.ndarray): Water-dependent hatching rate array.

    Returns:
        np.ndarray: Array of derivatives for each compartment, representing the rates of change.
    """

    # Unpack variables
    (
        t_idx,
        step_t,
        CC,
        birth,
        dia_lay,
        dia_hatch,
        mort_e,
        mort_j,
        mort_a,
        ed_surv,
        dev_j,
        dev_i,
        dev_e,
        water_hatch,
    ) = model_params

    # Initialize output array
    derivatives = np.zeros_like(state)

    logger.debug(f"dimension FT[...,0]: {derivatives[..., 0].shape}")

    # Differential equations (vectorized over grid)
    # 1. Egg compartment (non-diapause)
    derivatives[..., 0] = (
        state[..., 4] * birth * (1 - dia_lay)
        - (mort_e + (water_hatch * dev_e)) * state[..., 0]
    )

    # 2. Egg compartment (diapause)
    derivatives[..., 1] = (
        state[..., 4] * birth * dia_lay
        - water_hatch * dia_hatch * state[..., 1]  # Hatching from diapause
    )

    # 3. Juvenile compartment
    derivatives[..., 2] = (
        water_hatch * dev_e * state[..., 0]  # Hatching from non-diapause eggs
        + water_hatch
        * dia_hatch
        * ed_surv
        * state[..., 1]  # Hatching from diapause eggs
        - (mort_j + dev_j) * state[..., 2]  # Mortality and development
        - (state[..., 2] ** 2) / CC[..., t_idx - 1]  # Density-dependent mortality
    )

    # 4. Immature adult compartment
    derivatives[..., 3] = (
        0.5 * dev_j * state[..., 2]  # Development from juveniles
        - (mort_a + dev_i) * state[..., 3]  # Mortality and maturation
    )

    # 5. Mature adult compartment
    derivatives[..., 4] = (
        dev_i * state[..., 3]  # Maturation from immature adults
        - mort_a * state[..., 4]  # Adult mortality
    )

    # Replace internal NaNs
    derivatives[np.isnan(-derivatives)] = -state[np.isnan(-derivatives)] * step_t

    return derivatives


def albopictus_log_ode_system(
    state: np.ndarray,
    model_params: tuple[
        int,  # time_index
        float,  # time_step
        np.ndarray,  # carrying_capacity
        np.ndarray,  # birth_rate
        np.ndarray,  # diapause_laying_fraction
        np.ndarray,  # diapause_hatching_fraction
        np.ndarray,  # egg_mortality
        np.ndarray,  # juvenile_mortality
        np.ndarray,  # adult_mortality
        np.ndarray,  # egg_diapause_survival
        np.ndarray,  # juvenile_development
        np.ndarray,  # immature_development
        float,  # egg_development
        np.ndarray,  # water_hatching_rate
    ],
) -> np.ndarray:
    """
    Computes the derivatives of the mosquito Aedes albopictus population compartments in log-transformed space for the ODE system.

    This function calculates the rates of change for each compartment (non-diapause eggs, diapause eggs,
    juveniles, immature adults, and mature adults) in log space, which helps to handle negative or very small values
    during numerical integration.

    Args:
        state (np.ndarray): Current state array of the system in log-transformed space, representing the population in each compartment.
        model_params (tuple): Tuple containing model parameters and environmental variables in the following order:
            time_index (int): Current time index.
            time_step (float): Time step size.
            carrying_capacity (np.ndarray): Carrying capacity array.
            birth_rate (np.ndarray): Birth rate array.
            diapause_laying_fraction (np.ndarray): Fraction of eggs laid in diapause.
            diapause_hatching_fraction (np.ndarray): Fraction of diapause eggs hatching.
            egg_mortality (np.ndarray): Egg mortality rate array.
            juvenile_mortality (np.ndarray): Juvenile mortality rate array.
            adult_mortality (np.ndarray): Adult mortality rate array.
            egg_diapause_survival (np.ndarray): Survival rate of diapause eggs.
            juvenile_development (np.ndarray): Juvenile development rate array.
            immature_development (np.ndarray): Immature adult development rate array.
            egg_development (float): Egg development rate (scalar).
            water_hatching_rate (np.ndarray): Water-dependent hatching rate array.

    Returns:
        np.ndarray: Array of derivatives for each compartment in log-transformed space, representing the rates of change.
    """

    # Unpack variables
    (
        t_idx,
        step_t,
        CC,
        birth,
        dia_lay,
        dia_hatch,
        mort_e,
        mort_j,
        mort_a,
        ed_surv,
        dev_j,
        dev_i,
        dev_e,
        water_hatch,
    ) = model_params

    # Initialize output array
    log_derivatives = np.zeros_like(state)

    # 1. Egg compartment (non-diapause)
    log_derivatives[..., 0] = state[..., 4] * birth * (1 - dia_lay) / state[..., 0] - (
        mort_e + water_hatch * dev_e
    )

    # 2. Egg compartment (diapause)
    log_derivatives[..., 1] = (
        state[..., 4] * birth * dia_lay / state[..., 1] - water_hatch * dia_hatch
    )

    # 3. Juvenile compartment
    log_derivatives[..., 2] = (
        water_hatch * dev_e * state[..., 0] / state[..., 2]
        + water_hatch * dia_hatch * ed_surv * state[..., 1] / state[..., 2]
        - (mort_j + dev_j)
        - state[..., 2] / CC[..., t_idx - 1]
    )

    # 4. Immature adult compartment
    log_derivatives[..., 3] = 0.5 * dev_j * state[..., 2] / state[..., 3] - (
        mort_a + dev_i
    )

    # 5. Mature adult compartment
    log_derivatives[..., 4] = dev_i * state[..., 3] / state[..., 4] - mort_a

    log_derivatives[np.isnan(-log_derivatives)] = (
        -state[np.isnan(-log_derivatives)] * step_t
    )

    return log_derivatives


def rk4_step(
    ode_func: Callable[[np.ndarray, tuple], np.ndarray],
    log_ode_func: Callable[[np.ndarray, tuple], np.ndarray],
    state: np.ndarray,
    model_params: tuple,
    time_step: float,
) -> np.ndarray:
    """
    Perform a single Runge-Kutta 4th order (RK4) integration step with negative value correction.

    Advances the system state by one time step using the RK4 method. If any negative values
    are detected in the output or intermediate steps, a log-transformed ODE system is used
    to correct those values.

    Args:
        ode_func (Callable[[np.ndarray, Tuple], np.ndarray]): Function that computes the ODE derivatives in natural scale.
        log_ode_func (Callable[[np.ndarray, Tuple], np.ndarray]): Function that computes the ODE derivatives in log scale.
        state (np.ndarray): Current state vector or array of the system.
        model_params (Tuple): Model parameters and environmental inputs required by the ODE functions.
        time_step (float): Time step size for integration.

    Returns:
        np.ndarray: Updated state vector or array after one RK4 integration step.
    """

    # Octave-style RK4 with negative value correction using log-form ODEs
    # k1, k2, k3, k4 computations follow the RK4 notation.

    k1 = ode_func(state, model_params)
    logger.debug(f"k1 min: {np.min(k1)}, max: {np.max(k1)}")

    k2 = ode_func(state + 0.5 * k1 / time_step, model_params)
    logger.debug(f"k2 min: {np.min(k2)}, max: {np.max(k2)}")

    k3 = ode_func(state + 0.5 * k2 / time_step, model_params)
    logger.debug(f"k3 min: {np.min(k3)}, max: {np.max(k3)}")

    k4 = ode_func(state + k3 / time_step, model_params)
    logger.debug(f"k4 min: {np.min(k4)}, max: {np.max(k4)}")

    rk4_step_out_array = state + (k1 + 2 * k2 + 2 * k3 + k4) / (time_step * 6.0)

    logger.debug(
        f"RK4 step min: {np.min(rk4_step_out_array)}, max: {np.max(rk4_step_out_array)}"
    )

    # Check for negative values in all RK4 steps
    neg_mask = (
        (rk4_step_out_array < 0)
        | ((state + 0.5 * k1 / time_step) < 0)
        | ((state + 0.5 * k2 / time_step) < 0)
        | ((state + k3 / time_step) < 0)
    )

    if np.any(neg_mask):
        v2 = np.log(np.clip(state, 1e-26, None))  # clip to avoid ~log(0)
        FT2 = log_ode_func(v2, model_params)
        v2 = v2 + FT2 / time_step
        rk4_step_out_array[neg_mask] = np.exp(v2[neg_mask], dtype=np.float64)

    return rk4_step_out_array


def solve_system(
    state: xr.DataArray | np.ndarray,
    temperature: np.ndarray,
    temperature_mean: np.ndarray,
    latitudes: np.ndarray,
    carrying_capacity: xr.DataArray,
    egg_activate: xr.DataArray,
    time_step: float,
) -> np.ndarray:
    """
    Runs the ODE solver for the mosquito population model over a time series of temperature and environmental data.

    This function iterates over time steps, computes all necessary model rates and parameters,
    and applies the Runge-Kutta 4th order (RK4) integration method to update the population state.
    The results for each compartment and time step are stored in the output array.

    Args:
        state (np.ndarray): Initial state array representing the population in each compartment.
        temperature (np.ndarray): 3D array of temperature values over space and time.
        temperature_mean (np.ndarray): 3D array of mean temperature values over space and time.
        latitudes (np.ndarray): Array of latitude values for each spatial location.
        carrying_capacity (xr.DataArray): Carrying capacity for each spatial location and time.
        egg_activate (xr.DataArray): Egg activation rates for each spatial location and time.
        time_step (float): Time step size for the integration.

    Returns:
        np.ndarray: 4D array containing the population state for each compartment, spatial location, and time step.
    """
    # Keep xarray at the data layer and run the RK4 core on one numeric type.
    if isinstance(state, xr.DataArray):
        state = state.values
    state = np.asarray(state, dtype=np.float64)

    # Precompute temperature-dependent rates
    diapause_lay = mosq_dia_lay(temperature_mean, latitudes)
    diapause_hatch = mosq_dia_hatch(temperature_mean, latitudes)
    ed_survival = mosq_surv_ed(temperature, time_step)

    v_out = Pmodel_initial.create_model_output(
        dataset_base_shape=temperature.shape,
        initial_conditions_shape=state.shape,
        time_step=time_step,
    )

    logger.debug(f"Shape v_out:{v_out.shape}")

    # Time loop
    for t in range(temperature.shape[2]):
        logger.debug(f"-----    Time step: {t}    -----")

        # if t == 3: # DEBUGGING ONLY
        #    break

        # Extract temperature slice and compute rates
        T = temperature[:, :, t]
        birth = mosq_birth(T)
        dev_j = mosq_dev_j(T)
        dev_i = mosq_dev_i(T)
        dev_e = np.array([1.0 / 7.1])  # original model

        # Octave: ceil(t/step_t), Python: int(np.ceil((t+1)/step_t)) - 1
        idx_time = int(np.ceil((t + 1) / time_step)) - 1

        # Extract time-dependent parameters
        dia_lay = diapause_lay.values[:, :, idx_time]

        # diapause_hatch at current time index
        dia_hatch = diapause_hatch.values[:, :, idx_time]

        #  egg_diapause_survival at current time index
        ed_surv = ed_survival[:, :, t]

        # water_hatch at current time index
        water_hatch = egg_activate.values[:, :, idx_time]

        # egg_mortality at current time index
        mort_e = mosq_mort_e(T)

        # juvenile_mortality at current time index
        mort_j = mosq_mort_j(T)

        temperature_mean_slice = temperature_mean[:, :, idx_time]
        logger.debug(f"Tmean shape: {temperature_mean.shape}")
        logger.debug(f"Tmean_slice shape: {temperature_mean_slice.shape}")

        # adult_mortality at current time index
        mort_a = mosq_mort_a(temperature_mean_slice)

        # Dimension checks for debugging
        logger.debug(f"Time step {t}:")
        logger.debug(f"  T.shape: {T.shape}")
        logger.debug(f"  birth.shape: {getattr(birth, 'shape', None)}")
        logger.debug(f"  dia_lay.shape: {getattr(dia_lay, 'shape', None)}")
        logger.debug(f"  mort_e.shape: {getattr(mort_e, 'shape', None)}")
        logger.debug(f"  water_hatch.shape: {getattr(water_hatch, 'shape', None)}")
        logger.debug(f"  dev_e: {dev_e}")
        logger.debug(f"  v[..., 0].shape: {state[..., 0].shape}")

        # Pack model parameters
        model_params = (
            idx_time + 1,  # Octave uses 1-based, so pass idx_time+1
            time_step,
            carrying_capacity.values,
            birth.values,
            dia_lay,
            dia_hatch,
            mort_e.values,
            mort_j.values,
            mort_a.values,
            ed_surv.values,
            dev_j.values,
            dev_i.values,
            dev_e,
            water_hatch,
        )

        # Calculate next state using RK4 step
        state = rk4_step(
            albopictus_ode_system,
            albopictus_log_ode_system,
            state,
            model_params,
            time_step,
        )

        # Reset diapause eggs to zero on day 200 of each year (to mimic seasonal behavior)
        if (t / time_step) % 365 == 200:  # Mid-July approx., assuming time_step in days
            state[..., 1] = 0

        # Store results at specified time steps
        if (t + 1) % time_step == 0:  # Store every 'time_step' steps
            logger.debug(f"Storing results at time step: {t}")
            if ((idx_time) % 30) == 0:  # Trigger every 30 time steps (approx. monthly)
                logger.debug(f"Month Of Year: {int(((t) / time_step) / 30)}")
            # Store non-negative values only in all the compartments
            for j in range(v_out.shape[2]):
                v_out[..., j, idx_time] = np.maximum(state[..., j], 0)

    return v_out
