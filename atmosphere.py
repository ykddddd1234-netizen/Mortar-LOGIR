import numpy as np

from constants import (
    rho0,
    H,
    gamma,
    R
)


# =========================================
# Density
# =========================================

def get_density(z):

    return rho0 * np.exp(
        -z / H
    )


# =========================================
# Temperature
# =========================================

def get_temperature(z):

    T0 = 288.15       # K
    lapse = 0.0065    # K/m

    return T0 - lapse * z


# =========================================
# Speed of Sound
# =========================================

def get_speed_of_sound(z):

    T = get_temperature(z)

    return np.sqrt(
        gamma * R * T
    )