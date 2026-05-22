import numpy as np
import matplotlib.pyplot as plt

from rk4 import rk4_step
from derivatives import derivatives

from constants import (
    g,
    m
)

from units import (
    mil_to_deg,
    deg_to_mil,
    mil_to_rad,
    rad_to_mil
)


# =========================
# Initial Conditions
# =========================

v0 = 259.3

theta_mil = 1511
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
        derivatives
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
# Energy Check
# =========================

E = (
    0.5 * m * V**2
    +
    m * g * z
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

print ("Initial Velocity  :", v0, "m/s")
print ("elevation angle   :", theta_mil, "mil")

print("Initial Energy :", E[0])
print("Final Energy   :", E[-1])

print("Energy Loss    :", E[0] - E[-1])

print("Range          :", x[-1], "m")

print("Flight Time    :", time, "s")

print("Max Altitude   :", np.max(z), "m")

print("Impact Velocity:", V[-1], "m/s")

print("Impact Angle :", deg_to_mil(impact_angle), "mil")

# =========================
# Plot Trajectory
# =========================

plt.figure(figsize=(10,6))

plt.plot(x, z)

plt.xlabel("Range [m]")
plt.ylabel("Altitude [m]")

plt.title("Trajectory with Drag")

plt.grid(True)

plt.savefig("trajectory.png")