import numpy as np

from rk4 import rk4_step

from derivatives import derivatives

from units import (

    deg_to_mil,

    mil_to_rad,
)

from aero import build_cd_interp


# =========================================
# Simulation
# =========================================

def simulate(

    v0,

    theta_mil,

    cd_table,

    return_trajectory=False
):

    # =====================================
    # Launch Angles
    # =====================================

    interp = build_cd_interp(cd_table)

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
    # Initial State Vector
    # [x, y, z, vx, vy, vz]
    # =====================================

    state = np.array([

        0.0,
        0.0,
        0.0,

        vx0,
        vy0,
        vz0
    ])


    # =====================================
    # Simulation Settings
    # =====================================

    dt = 0.05

    time = 0.0

    max_z = 0.0


    # =====================================
    # Trajectory Storage
    # =====================================

    trajectory = []


    # =====================================
    # Simulation Loop
    # =====================================

    while state[2] >= 0:


        # =================================
        # Save Trajectory
        # =================================

        if return_trajectory:

            trajectory.append(state.copy())


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

            lambda s: derivatives(s, interp)
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


    # =====================================
    # Add Trajectory
    # =====================================

    if return_trajectory:

        result["trajectory"] = np.array(trajectory)


    return result