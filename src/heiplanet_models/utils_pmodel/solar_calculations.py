import logging

import numpy as np

from heiplanet_models.Pmodel.Pmodel_params import (
    CONSTANT_DECLINATION_ANGLE,
    CONSTANTS_REVOLUTION_ANGLE,
    HOURS_PER_DAY,
    MAX_LAT_DEGREES,
    MIN_LAT_DEGREES,
)

# ---- Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---- General functions
def revolution_angle(days: int | np.ndarray) -> float | np.ndarray:
    """Calculate Earth's revolution angle in radians for given days of the year.

    This function can process both a single integer day or a numpy array of days.

    Args:
        days (Union[int, np.ndarray]): Day or days of the year (values must be
            in the range 1-366, inclusive).

    Returns:
        Union[float, np.ndarray]: Revolution angle or angles in radians.
            Returns a float for a single day input, and a numpy array for an
            array input.

    Raises:
        TypeError: If `days` is not an integer or an array of integers.
        ValueError: If any day is not in the range 1 to 366.
    """

    CONST_1 = CONSTANTS_REVOLUTION_ANGLE["CONST_1"]
    CONST_2 = CONSTANTS_REVOLUTION_ANGLE["CONST_2"]
    CONST_3 = CONSTANTS_REVOLUTION_ANGLE["CONST_3"]
    CONST_4 = CONSTANTS_REVOLUTION_ANGLE["CONST_4"]
    CONST_5 = CONSTANTS_REVOLUTION_ANGLE["CONST_5"]

    days_arr = np.asarray(days)

    if not np.issubdtype(days_arr.dtype, np.integer):
        raise TypeError("Input 'days' must be an integer or an array of integers.")

    if not np.all((days_arr >= 1) & (days_arr <= 366)):
        raise ValueError("All 'days' must be in the range 1 to 366.")

    theta = CONST_1 + 2 * np.arctan(
        CONST_2 * np.tan(CONST_3 * ((days_arr % CONST_4) - CONST_5))
    )

    return theta.item() if days_arr.ndim == 0 else theta


def declination_angle(revolution_angle: float) -> float:
    """Predict the sun's declination angle in radians.

    Args:
        revolution_angle (float): Revolution angle in radians.

    Returns:
        float: Sun's declination angle in radians.
    """

    phi = np.arcsin(CONSTANT_DECLINATION_ANGLE * np.cos(revolution_angle))
    return phi


def daylight_forsythe(
    latitude: float, declination_angle: float, daylight_coefficient: float
) -> float:
    """Calculate daylight hours using Forsythe's method.

    Args:
        latitude (float): Latitude in degrees (-90 to 90).
        declination_angle (float): Declination angle in radians.
        daylight_coefficient (float): Daylight coefficient in degrees.

    Returns:
        float: Daylight hours.

    Raises:
        ValueError: If latitude is not between -90 and 90 degrees.
    """

    if not MIN_LAT_DEGREES <= latitude <= MAX_LAT_DEGREES:
        raise ValueError("Latitude must be between -90 and 90 degrees.")

    latitude_rad = np.deg2rad(latitude)
    daylight_coeff_rad = np.deg2rad(daylight_coefficient).squeeze()

    angle_calculation = (
        np.sin(daylight_coeff_rad) + np.sin(latitude_rad) * np.sin(declination_angle)
    ) / (np.cos(latitude_rad) * np.cos(declination_angle))

    daylight = np.real(
        HOURS_PER_DAY - (HOURS_PER_DAY / np.pi) * np.arccos(angle_calculation + 0j)
    )
    return daylight
