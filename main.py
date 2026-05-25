import numpy as np
import matplotlib.pyplot as plt

from rk4 import rk4_step
from derivatives import derivatives

from units import (
    deg_to_mil,
    mil_to_rad,
)

from aero import (

    mach_table,

    build_cd_interp,

    get_cd
)

from atmosphere import (
    get_speed_of_sound
)

from optimized_cd_tables import (

    cd_table_3,

    cd_table_4,

    cd_table_5,

    cd_table_6
)


# =========================================
# Charge-dependent Settings
# =========================================

charge_settings = {

    1: {

        "v0": 130.8,

        "cd_table": cd_table_3
    },

    2: {

        "v0": 179.9,

        "cd_table": cd_table_3
    },

    3: {

        "v0": 220.4,

        "cd_table": cd_table_3
    },

    4: {

        "v0": 262.0,

        "cd_table": cd_table_4
    },

    5: {

        "v0": 288.2,

        "cd_table": cd_table_5
    },

    6: {

        "v0": 320.0,

        "cd_table": cd_table_6
    }
}


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
    # Build Interpolator
    # =====================================

    interp = build_cd_interp(cd_table)


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
    # Initial State Vector
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

    dt = 0.01

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

            vx = state[3]

            vy = state[4]

            vz = state[5]


            # =============================
            # Velocity Magnitude
            # =============================

            V = np.sqrt(

                vx**2 +

                vy**2 +

                vz**2
            )


            # =============================
            # Current Altitude
            # =============================

            z = state[2]


            # =============================
            # Speed of Sound
            # =============================

            a = get_speed_of_sound(z)


            # =============================
            # Mach Number
            # =============================

            M = V / a


            # =============================
            # Current Cd
            # =============================

            Cd = get_cd(M, interp)


            # =============================
            # Save
            # =============================

            trajectory.append([

                time,

                state[0],   # x
                state[1],   # y
                state[2],   # z

                V,
                M,
                Cd
            ])


        # =================================
        # Maximum Altitude
        # =================================

        if state[2] > max_z:

            max_z = state[2]


        # =================================
        # RK4 Integration
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

        trajectory = np.array(trajectory)

        result["trajectory"] = trajectory


    return result


# =========================================
# User Input
# =========================================

charge = int(

    input(

        "Charge Number (1~6): "
    )
)

theta_mil = float(

    input(

        "Elevation Angle (mil): "
    )
)


# =========================================
# Charge Validation
# =========================================

if charge not in charge_settings:

    raise ValueError(

        "Charge must be between 1 and 6."
    )


# =========================================
# Load Charge-dependent Data
# =========================================

v0 = charge_settings[charge]["v0"]

cd_table = charge_settings[charge]["cd_table"]


# =========================================
# Run Simulation
# =========================================

result = simulate(

    v0=v0,

    theta_mil=theta_mil,

    cd_table=cd_table,

    return_trajectory=True
)


# =========================================
# Print Results
# =========================================

print("\n====================================")
print("Simulation Result")
print("====================================\n")

print(

    "Charge           :",

    charge
)

print(

    "Muzzle Velocity  :",

    round(v0, 3),

    "m/s"
)

print(

    "Range            :",

    round(result["range"], 3),

    "m"
)

print(

    "Time of Flight   :",

    round(result["tof"], 3),

    "s"
)

print(

    "Maximum Altitude :",

    round(result["hmax"], 3),

    "m"
)

print(

    "Impact Velocity  :",

    round(result["impact_velocity"], 3),

    "m/s"
)

print(

    "Impact Angle     :",

    round(result["impact_angle"], 3),

    "mil"
)

print()


# =========================================
# Trajectory Extraction
# =========================================

traj = result["trajectory"]

time_hist = traj[:,0]

x = traj[:,1]

y = traj[:,2]

z = traj[:,3]

V_hist = traj[:,4]

M_hist = traj[:,5]

Cd_hist = traj[:,6]


# =========================================
# Save Trajectory Data
# =========================================

trajectory_data = np.column_stack((

    time_hist,

    x,

    y,

    z,

    V_hist,

    M_hist,

    Cd_hist
))

np.savetxt(

    "trajectory.txt",

    trajectory_data,

    header=(

        "time(s) "
        "x(m) "
        "y(m) "
        "z(m) "
        "V(m/s) "
        "Mach "
        "Cd"
    ),

    fmt="%.6f"
)


# =========================================
# Plot Trajectory
# =========================================

plt.figure(figsize=(10,5))

plt.plot(x, z)

plt.xlabel("Range (m)")

plt.ylabel("Altitude (m)")

plt.title(

    f"Mortar Trajectory - Charge {charge}"
)

plt.grid(True)

plt.savefig(

    "trajectory.png",

    dpi=300
)

plt.close()


# =========================================
# Done
# =========================================

print("Saved Files:")

print(" - trajectory.txt")

print(" - trajectory.png")