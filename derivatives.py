# derivatives.py

import numpy as np

from constants import (

    g,

    S,

    m
)

from aero import (

    get_cd,

    sigmoid
)

from atmosphere import (

    get_density,

    get_speed_of_sound
)


def derivatives(

    state,

    params
):

    # =====================================
    # State
    # =====================================

    x, y, z, vx, vy, vz, a = state


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
    # Parameters
    # =====================================

    Mc = params["Mc"]

    k = params["k"]

    tau = params["tau"]


    # =====================================
    # Target Activation
    # =====================================

    S_target = sigmoid(

        M,

        Mc,

        k
    )


    # =====================================
    # Activation Dynamics
    # =====================================

    da_dt = (

        S_target - a

    ) / tau


    # =====================================
    # Drag Coefficient
    # =====================================

    Cd = get_cd(

        M,

        a,

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


    return np.array([

        vx,

        vy,

        vz,

        ax,

        ay,

        az,

        da_dt
    ])