import numpy as np

from rk4 import rk4_step

from derivatives import derivatives

from atmosphere import (
    get_speed_of_sound
)

from units import mil_to_rad

from met import (
    set_uniform_density_ratio,
    set_uniform_temp_ratio
)

from aero import (
    get_cd,
    hump
)


# =========================================
# Run Case
# =========================================

def run_case(

    v0,

    theta_mil,

    density_scale,

    temp_scale,

    params
):

    # =====================================
    # MET
    # =====================================

    set_uniform_density_ratio(

        density_scale
    )

    set_uniform_temp_ratio(

        temp_scale
    )


    # =====================================
    # Launch Angle
    # =====================================

    theta = mil_to_rad(

        theta_mil
    )

    psi = 0.0


    # =====================================
    # Initial Velocity
    # =====================================

    vx0 = (

        v0

        * np.cos(theta)

        * np.cos(psi)
    )

    vy0 = (

        v0

        * np.cos(theta)

        * np.sin(psi)
    )

    vz0 = (

        v0

        * np.sin(theta)
    )


    # =====================================
    # Initial State
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
    # Histories
    # =====================================

    time_history = []

    mach_history = []

    cd_history = []

    hump_history = []

    range_history = []

    height_history = []


    # =====================================
    # Settings
    # =====================================

    dt = 0.005

    time = 0.0

    max_z = 0.0


    # =====================================
    # Simulation Loop
    # =====================================

    while state[2] >= 0:

        x, y, z, vx, vy, vz = state


        # =================================
        # Max Altitude
        # =================================

        if z > max_z:

            max_z = z


        # =================================
        # Velocity
        # =================================

        V = np.sqrt(

            vx**2

            + vy**2

            + vz**2
        )


        # =================================
        # Mach
        # =================================

        speed_of_sound = get_speed_of_sound(z)

        M = V / speed_of_sound


        # =================================
        # Cd
        # =================================

        Cd = get_cd(

            M,

            params
        )


        # =================================
        # Hump
        # =================================

        H = hump(

            M,

            params
        )


        # =================================
        # Save Histories
        # =================================

        time_history.append(time)

        mach_history.append(M)

        cd_history.append(Cd)

        hump_history.append(H)

        range_history.append(x)

        height_history.append(z)


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
    # Final Velocity
    # =====================================

    V_final = np.sqrt(

        vx_final**2

        + vy_final**2

        + vz_final**2
    )


    # =====================================
    # Impact Angle
    # =====================================

    V_horizontal = np.sqrt(

        vx_final**2

        + vy_final**2
    )

    impact_angle_deg = np.degrees(

        np.arctan2(

            abs(vz_final),

            V_horizontal
        )
    )

    impact_angle_mil = (

        impact_angle_deg

        * (6400.0 / 360.0)
    )


    # =====================================
    # Result
    # =====================================

    result = {

        "range": x_final,

        "tof": time,

        "hmax": max_z,

        "impact_velocity": V_final,

        "impact_angle": impact_angle_mil,

        "time_history": np.array(time_history),

        "mach_history": np.array(mach_history),

        "cd_history": np.array(cd_history),

        "hump_history": np.array(hump_history),

        "range_history": np.array(range_history),

        "height_history": np.array(height_history)
    }


    return result