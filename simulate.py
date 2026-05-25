# simulate.py

import numpy as np

from rk4 import rk4_step

from derivatives import derivatives

from units import (

    deg_to_mil,

    mil_to_rad
)


# =========================================
# Simulation
# =========================================

def simulate(

    v0,

    theta_mil,

    params
):

    # =====================================
    # Launch Angles
    # =====================================

    theta = mil_to_rad(theta_mil)

    psi_mil = 0

    psi = mil_to_rad(psi_mil)


    # =====================================
    # Initial Velocity Components
    # =====================================

    vx0 = v0 * np.cos(theta) * np.cos(psi)

    vy0 = v0 * np.cos(theta) * np.sin(psi)

    vz0 = v0 * np.sin(theta)


    # =====================================
    # Initial Activation
    # =====================================

    a0 = 0.0


    # =====================================
    # Initial State Vector
    # [x, y, z, vx, vy, vz, a]
    # =====================================

    state = np.array([

        0.0,
        0.0,
        0.0,

        vx0,
        vy0,
        vz0,

        a0
    ])


    # =====================================
    # Simulation Settings
    # =====================================

    dt = 0.05

    time = 0.0

    max_z = 0.0


    # =====================================
    # Simulation Loop
    # =====================================

    while state[2] >= 0:


        # =================================
        # Maximum Altitude
        # =================================

        if state[2] > max_z:

            max_z = state[2]


        # =================================
        # RK4 Step
        # =================================

        state = rk4_step(

            state,

            dt,

            lambda s: derivatives(

                s,

                params
            )
        )

        time += dt


    # =====================================
    # Final State
    # =====================================

    x_final = state[0]

    vx_final = state[3]

    vy_final = state[4]

    vz_final = state[5]


    # =====================================
    # Final Velocity Magnitude
    # =====================================

    V_final = np.sqrt(

        vx_final**2 +

        vy_final**2 +

        vz_final**2
    )


    # =====================================
    # Impact Angle
    # =====================================

    V_horizontal = np.sqrt(

        vx_final**2 +

        vy_final**2
    )

    impact_angle = np.degrees(

        np.arctan2(

            abs(vz_final),

            V_horizontal
        )
    )


    # =====================================
    # Result Dictionary
    # =====================================

    result = {

        "range": x_final,

        "tof": time,

        "hmax": max_z,

        "impact_velocity": V_final,

        "impact_angle": deg_to_mil(impact_angle)
    }


    return result