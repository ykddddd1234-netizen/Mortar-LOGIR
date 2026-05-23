import numpy as np
import matplotlib.pyplot as plt

from main import simulate
from cases import cases

from aero import mach_table


cd_table = np.array([

    # =========================================
    # Subsonic
    # =========================================

    0.130,  # M = 0.00
    0.130,  # M = 0.30
    0.130,  # M = 0.50
    0.130,  # M = 0.70

    # =========================================
    # Transonic
    # =========================================

    0.130,  # M = 0.74
    0.138,  # M = 0.76

    0.150,  # M = 0.78
    0.165,  # M = 0.80
    0.174,  # M = 0.82
    0.170,  # M = 0.84
    0.152,  # M = 0.86

    0.138,  # M = 0.88
    0.132,  # M = 0.90

    # =========================================
    # Supersonic
    # =========================================

    0.129,  # M = 1.00
    0.127,  # M = 1.10
    0.125   # M = 1.20
])


# =========================================
# Test Cases
# =========================================

for case in cases:

    sim = simulate(

        v0=case["v0"],

        theta_mil=case["theta_mil"],

        cd_table=cd_table
    )

    print("====================================")
    print(case["name"])
    print("====================================")

    print(
        "Range :",
        sim["range"],
        "| target:",
        case["range"]
    )

    print(
        "TOF :",
        sim["tof"],
        "| target:",
        case["tof"]
    )

    print(
        "Impact Angle :",
        sim["impact_angle"],
        "| target:",
        case["impact_angle"]
    )

    print(
        "Impact Velocity :",
        sim["impact_velocity"],
        "| target:",
        case["impact_velocity"]
    )

    print(
        "Hmax :",
        sim["hmax"],
        "| target:",
        case["hmax"]
    )

    print()

# =========================================
# Cd(M) Plot
# =========================================

plt.figure(figsize=(8,5))

plt.plot(

    mach_table,

    cd_table,

    marker='o'
)

plt.ticklabel_format(style='plain')

plt.xlabel("Mach Number")

plt.ylabel("Cd")

plt.title("Test Cd(M) Table")

plt.grid(True)

plt.savefig("cd_curve.png")

plt.show()