# simulate.py

import numpy as np

import matplotlib.pyplot as plt

from main import run_case


# =========================================
# Optimized Parameters
# =========================================

params = {

    "C0": 0.142,

    "C1": -0.010,

    "A": 0.552486,

    "M1": 0.942764,

    "M2": 1.085635,

    "k1": 149.999543,

    "k2": 7.021692
}


# =========================================
# Simulation Settings
# =========================================

v0 = 320

theta_mil = 1067


# =========================================
# Run Cases
# =========================================

nominal = run_case(

    v0=v0,

    theta_mil=theta_mil,

    density_scale=1.0,

    temp_scale=1.0,

    params=params
)

warm = run_case(

    v0=v0,

    theta_mil=theta_mil,

    density_scale=1.0,

    temp_scale=1.01,

    params=params
)

cold = run_case(

    v0=v0,

    theta_mil=theta_mil,

    density_scale=1.0,

    temp_scale=0.99,

    params=params
)


# =========================================
# Corrections
# =========================================

warm_corr = (

    warm["range"]

    - nominal["range"]
)

cold_corr = (

    cold["range"]

    - nominal["range"]
)


# =========================================
# Print Result
# =========================================

print("\n========================================")
print("SIMULATION RESULT")
print("========================================\n")

print(

    f"Nominal Range       : "

    f"{nominal['range']:.2f} m"
)

print(

    f"Temp +1% Range      : "

    f"{warm['range']:.2f} m"
)

print(

    f"Temp -1% Range      : "

    f"{cold['range']:.2f} m"
)

print()

print(

    f"Temp +1% Correction : "

    f"{warm_corr:.2f} m"
)

print(

    f"Temp -1% Correction : "

    f"{cold_corr:.2f} m"
)

print()

print(

    f"|Cold| / |Warm|     : "

    f"{abs(cold_corr)/abs(warm_corr):.3f}"
)


# =========================================
# Plot
# =========================================

fig, ax = plt.subplots(

    figsize=(10, 6)
)


# =========================================
# Cd vs Mach
# =========================================

ax.plot(

    nominal["mach_history"],

    nominal["cd_history"],

    label="Nominal"
)

ax.plot(

    warm["mach_history"],

    warm["cd_history"],

    label="Temp +1%"
)

ax.plot(

    cold["mach_history"],

    cold["cd_history"],

    label="Temp -1%"
)


# =========================================
# Labels
# =========================================

ax.set_xlabel("Mach")

ax.set_ylabel("Cd")

ax.set_title("Transonic Drag Rise")


# =========================================
# Grid / Legend
# =========================================

ax.grid(True)

ax.legend()


# =========================================
# Layout
# =========================================

plt.tight_layout()


# =========================================
# Save
# =========================================

plt.savefig(

    "transonic_drag.png",

    dpi=300
)

print()

print("Saved : transonic_drag.png")


# =========================================
# Show
# =========================================

plt.show()