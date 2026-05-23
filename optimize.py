import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from main import simulate
from cases import cases

from aero import mach_table


# =========================================
# Objective Function
# =========================================

def objective(params):

    (
        Cd_076,
        Cd_078,
        Cd_080,
        Cd_082,
        Cd_084
    ) = params


    # =========================================
    # Aero Table
    # =========================================

    cd_table = np.array([

        # ---------------------------------
        # Subsonic
        # ---------------------------------

        0.130,  # M = 0.00
        0.130,  # M = 0.30
        0.130,  # M = 0.50
        0.130,  # M = 0.70

        # ---------------------------------
        # Transonic
        # ---------------------------------

        0.130,   # M = 0.74

        Cd_076,  # M = 0.76
        Cd_078,  # M = 0.78
        Cd_080,  # M = 0.80
        Cd_082,  # M = 0.82
        Cd_084,  # M = 0.84

        0.130,   # M = 0.86
        0.130,   # M = 0.88
        0.130,   # M = 0.90

        # ---------------------------------
        # Supersonic
        # ---------------------------------

        0.130,   # M = 1.00
        0.130,   # M = 1.10
        0.130    # M = 1.20
    ])


    total_error = 0.0


    # =========================================
    # Case Loop
    # =========================================

    for case in cases:

        result = simulate(

            v0=case["v0"],

            theta_mil=case["theta_mil"],

            cd_table=cd_table
        )


        # =========================================
        # Relative Errors
        # =========================================

        range_error = (
            (result["range"] - case["range"])
            / case["range"]
        )

        tof_error = (
            (result["tof"] - case["tof"])
            / case["tof"]
        )

        angle_error = (
            (
                result["impact_angle"]
                - case["impact_angle"]
            )
            / case["impact_angle"]
        )

        velocity_error = (
            (
                result["impact_velocity"]
                - case["impact_velocity"]
            )
            / case["impact_velocity"]
        )

        hmax_error = (
            (result["hmax"] - case["hmax"])
            / case["hmax"]
        )


        # =========================================
        # Weighted Error
        # =========================================

        total_error += (

            5.0 * range_error**2 +

            2.0 * tof_error**2 +

            2.0 * angle_error**2 +

            1.0 * velocity_error**2 +

            1.0 * hmax_error**2
        )

    return total_error


# =========================================
# Initial Guess
# =========================================

initial_guess = [

    0.13,  # M = 0.76
    0.1682,  # M = 0.78
    0.2032,  # M = 0.80
    0.1877,  # M = 0.82
    0.13   # M = 0.84
]


# =========================================
# Optimization
# =========================================

result = minimize(

    objective,

    initial_guess,

    method="L-BFGS-B",

    bounds=[

        (0.130, 0.160),  # M = 0.76

        (0.140, 0.180),  # M = 0.78

        (0.160, 0.220),  # M = 0.80

        (0.140, 0.200),  # M = 0.82

        (0.130, 0.180)   # M = 0.84
    ],

    options={

        "maxiter": 30
    }
)


# =========================================
# Optimized Parameters
# =========================================

(
    Cd_076_opt,
    Cd_078_opt,
    Cd_080_opt,
    Cd_082_opt,
    Cd_084_opt
) = result.x


# =========================================
# Final Aero Table
# =========================================

cd_table = np.array([

    # ---------------------------------
    # Subsonic
    # ---------------------------------

    0.130,
    0.130,
    0.130,
    0.130,

    # ---------------------------------
    # Transonic
    # ---------------------------------

    0.130,

    Cd_076_opt,
    Cd_078_opt,
    Cd_080_opt,
    Cd_082_opt,
    Cd_084_opt,

    0.140,
    0.132,
    0.130,

    # ---------------------------------
    # Supersonic
    # ---------------------------------

    0.130,
    0.130,
    0.130
])


# =========================================
# Results
# =========================================

print("\n===== Optimization Result =====\n")

print("Cd @ M=0.76 :", Cd_076_opt)
print("Cd @ M=0.78 :", Cd_078_opt)
print("Cd @ M=0.80 :", Cd_080_opt)
print("Cd @ M=0.82 :", Cd_082_opt)
print("Cd @ M=0.84 :", Cd_084_opt)

print("\nTotal Error :", result.fun)

print("\n")


# =========================================
# Validation
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

plt.title("Optimized Cd(M) Aero Table")

plt.grid(True)

plt.savefig("cd_curve.png")