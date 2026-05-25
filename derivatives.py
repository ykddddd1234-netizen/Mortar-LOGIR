import numpy as np

from constants import (
    g,
    S,
    m,
)

from aero import get_cd
from atmosphere import (
    get_density,
    get_speed_of_sound
)


def derivatives(state, params):

    x, y, z, vx, vy, vz = state

    rho = get_density(z)


    # =========================
    # Velocity Magnitude
    # =========================

    V = np.sqrt(
        vx**2 +
        vy**2 +
        vz**2
    )

    # =========================
    # Drag Coefficient
    # =========================

    a = get_speed_of_sound(z)

    M = V / a

    Cd = get_cd(M, params)

    # =========================
    # Drag Magnitude
    # =========================

    D = (
        0.5
        * rho
        * V**2
        * Cd
        * S
    )


    # =========================
    # Drag Components
    # =========================

    Dx = -D * (vx / V)
    Dy = -D * (vy / V)
    Dz = -D * (vz / V)


    # =========================
    # Accelerations
    # =========================

    ax = Dx / m
    ay = Dy / m
    az = Dz / m - g


    return np.array([
        vx,
        vy,
        vz,
        ax,
        ay,
        az
    ])