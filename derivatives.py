# derivatives.py

import numpy as np

from constants import (

    g,

    S,

    m
)

from atmosphere import (

    get_density,

    get_speed_of_sound
)

from aero import get_cd


# =========================================
# Derivatives
# =========================================

def derivatives(

    state,

    params
):

    # =====================================
    # State
    # =====================================

    x, y, z, vx, vy, vz = state


    # =====================================
    # Atmosphere
    # =====================================

    rho = get_density(z)

    speed_of_sound = get_speed_of_sound(z)


    # =====================================
    # Velocity Magnitude
    # =====================================

    V = np.sqrt(

        vx**2 +

        vy**2 +

        vz**2
    )


    # =====================================
    # Mach Number
    # =====================================

    M = V / speed_of_sound


    # =====================================
    # Cd
    # =====================================

    Cd = get_cd(

        M,

        params
    )


    # =====================================
    # Drag Magnitude
    # =====================================

    D = (

        0.5

        * rho

        * V**2

        * Cd

        * S
    )


    # =====================================
    # Drag Components
    # =====================================

    Dx = -D * (vx / V)

    Dy = -D * (vy / V)

    Dz = -D * (vz / V)


    # =====================================
    # Accelerations
    # =====================================

    ax = Dx / m

    ay = Dy / m

    az = Dz / m - g


    # =====================================
    # Return
    # =====================================

    return np.array([

        vx,

        vy,

        vz,

        ax,

        ay,

        az
    ])