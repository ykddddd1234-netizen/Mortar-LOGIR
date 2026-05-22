import numpy as np
import matplotlib.pyplot as plt

from rk4 import rk4_step
from derivatives import derivatives

from units import (
    deg_to_mil,
    mil_to_rad,
)


# =========================
# Simulation
# =========================


def simulate(v0, theta_mil, A, sigma, Mc):

    theta = mil_to_rad(theta_mil)

    psi_mil = 0
    psi = mil_to_rad(psi_mil)


    # Velocity Components

    vx0 = v0 * np.cos(theta) * np.cos(psi)
    vy0 = v0 * np.cos(theta) * np.sin(psi)
    vz0 = v0 * np.sin(theta)


    # Initial State Vector
    # [x, y, z, vx, vy, vz]

    state = np.array([
        0.0,
        0.0,
        0.0,
        vx0,
        vy0,
        vz0
    ])


    # =========================
    # Simulation Settings
    # =========================

    dt = 0.01

    trajectory = []

    time = 0.0


    # =========================
    # Simulation Loop
    # =========================

    while state[2] >= 0:

        trajectory.append(state.copy())

        state = rk4_step(
            state,
            dt,
            lambda s: derivatives(s, A, sigma, Mc)
        )

        time += dt


    trajectory = np.array(trajectory)


    # =========================
    # Extract States
    # =========================

    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]

    vx = trajectory[:, 3]
    vy = trajectory[:, 4]
    vz = trajectory[:, 5]


    # =========================
    # Velocity Magnitude
    # =========================

    V = np.sqrt(
        vx**2 +
        vy**2 +
        vz**2
    )

    # =========================
    # impact_angle
    # =========================

    vx_impact = vx[-1]
    vy_impact = vy[-1]
    vz_impact = vz[-1]

    V_horizontal = np.sqrt(
        vx_impact**2 +
        vy_impact**2
    )

    impact_angle = np.degrees(
        np.arctan2(
            abs(vz_impact),
            V_horizontal
        )
    )

    return {
        "range": x[-1],
        "tof": time,
        "hmax": np.max(z),
        "impact_velocity": V[-1],
        "impact_angle": deg_to_mil(impact_angle)
    }

if __name__ == "__main__":

    result = simulate(
        v0=259.3,
        theta_mil=1511,
        A=0.04,
        sigma=0.10,
        Mc=0.85
    )

    print(result)