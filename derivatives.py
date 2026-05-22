import numpy as np

from constants import (
    g,
    rho0,
    S,
    m,
    H,
    a
)


def derivatives(state, A, sigma, Mc):

    x, y, z, vx, vy, vz = state

    rho = rho0 * np.exp(-z / H)


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

    M = V / a

    Cd = 0.13 + A * np.exp(
        -((M - Mc)/sigma)**2
    )

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