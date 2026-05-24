import numpy as np

from constants import (
    gamma,
    R,
    g
)

# =========================================
# Sea Level Standard Atmosphere
# =========================================

T0 = 288.15        # K

P0 = 101325.0      # Pa

L = 0.0065         # K/m


# =========================================
# Temperature
# =========================================

def get_temperature(z):

    return T0 - L * z


# =========================================
# Pressure
# =========================================

def get_pressure(z):

    T = get_temperature(z)

    exponent = g / (R * L)

    return P0 * (

        T / T0

    ) ** exponent


# =========================================
# Density
# =========================================

def get_density(z):

    T = get_temperature(z)

    P = get_pressure(z)

    return P / (R * T)


# =========================================
# Speed of Sound
# =========================================

def get_speed_of_sound(z):

    T = get_temperature(z)

    return np.sqrt(

        gamma * R * T
    )