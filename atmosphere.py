import numpy as np

from constants import (
    gamma,
    R,
    g
)

from met import (
    get_met_ratios
)


# =========================================
# Sea Level Standard Atmosphere
# =========================================

T0 = 288.15

P0 = 101325.0

L = 0.0065


# =========================================
# Standard Temperature
# =========================================

def get_standard_temperature(z):

    return T0 - L * z


# =========================================
# Standard Pressure
# =========================================

def get_standard_pressure(z):

    T = get_standard_temperature(z)

    exponent = g / (R * L)

    return P0 * (

        T / T0

    ) ** exponent


# =========================================
# Standard Density
# =========================================

def get_standard_density(z):

    T = get_standard_temperature(z)

    P = get_standard_pressure(z)

    return P / (R * T)


# =========================================
# Actual Density
# =========================================

def get_density(z):

    rho_std = get_standard_density(z)

    density_ratio, _ = get_met_ratios(z)

    return rho_std * density_ratio


# =========================================
# Actual Temperature
# =========================================

def get_temperature(z):

    T_std = get_standard_temperature(z)

    _, temp_ratio = get_met_ratios(z)

    return T_std * temp_ratio


# =========================================
# Speed of Sound
# =========================================

def get_speed_of_sound(z):

    T = get_temperature(z)

    return np.sqrt(

        gamma * R * T
    )