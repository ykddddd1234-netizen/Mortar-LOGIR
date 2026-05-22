import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from main import simulate
from cases import cases


# =========================================
# Objective Function
# =========================================

def objective(params):

    A, sigma, Mc = params

    total_error = 0.0

    for case in cases:

        result = simulate(
            v0=case["v0"],
            theta_mil=case["theta_mil"],
            A=A,
            sigma = sigma,
            Mc = Mc
        )

        # =========================
        # Relative Errors
        # =========================

        range_error = (
            (result["range"] - case["range"])
            / case["range"]
        )

        tof_error = (
            (result["tof"] - case["tof"])
            / case["tof"]
        )

        angle_error = (
            (result["impact_angle"] - case["impact_angle"])
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

        # =========================
        # Weighted Sum
        # =========================

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

    0.03,   # A
    0.06,   # sigma
    0.83    # Mc
]


# =========================================
# Optimization
# =========================================

result = minimize(

    objective,

    initial_guess,

    method="L-BFGS-B",

    bounds=[

        (0.0, 0.08),   # A
        (0.04, 0.12),   # sigma
        (0.78, 0.90)   # Mc
    ]
)

# =========================================
# Results
# =========================================

A_opt, sigma_opt, Mc_opt = result.x

print("\n===== Optimization Result =====")

print("A      :", A_opt)
print("sigma  :", sigma_opt)
print("Mc     :", Mc_opt)

print("error  :", result.fun)

print("\n")


# =========================================
# Validation
# =========================================

for case in cases:

    sim = simulate(

        v0=case["v0"],
        theta_mil=case["theta_mil"],

        A=A_opt,
        sigma=sigma_opt,
        Mc=Mc_opt
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

M = np.linspace(0, 1.2, 500)

Cd = 0.13 + A_opt * np.exp(
    -((M - Mc_opt)/sigma_opt)**2
)

plt.figure(figsize=(8,5))

plt.plot(M, Cd)

plt.xlabel("Mach Number")
plt.ylabel("Cd")

plt.title("Optimized Cd(M)")

plt.grid(True)

plt.savefig("cd_curve.png")