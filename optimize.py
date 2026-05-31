# optimize.py

import numpy as np

from scipy.optimize import minimize

from main import run_case

from cases import cases


# =========================================
# Objective Function
# =========================================

def objective(x):

    # =====================================
    # Parameters
    # =====================================

    params = {

        "C0": 0.128794,

        "C1": -0.005527,

        "C2": 0.010791,

        "A":  x[0],

        "M1": x[1],

        "M2": 1.1,

        "k1": x[2],

        "k2": 7.0
    }


    # =====================================
    # Total Loss
    # =====================================

    total_loss = 0.0


    # =====================================
    # Loop Over Cases
    # =====================================

    for case in cases:

        if not case.get("active", True):

            continue


        # =================================
        # Nominal
        # =================================

        nominal = run_case(

            v0=case["v0"],

            theta_mil=case["theta_mil"],

            density_scale=1.0,

            temp_scale=1.0,

            params=params
        )


        # =================================
        # Temperature +1%
        # =================================

        warm = run_case(

            v0=case["v0"],

            theta_mil=case["theta_mil"],

            density_scale=1.0,

            temp_scale=1.01,

            params=params
        )


        # =================================
        # Temperature -1%
        # =================================

        cold = run_case(

            v0=case["v0"],

            theta_mil=case["theta_mil"],

            density_scale=1.0,

            temp_scale=0.99,

            params=params
        )


        # =================================
        # Model Corrections
        # =================================

        temp_plus_model = (

            warm["range"]

            - nominal["range"]
        )

        temp_minus_model = (

            cold["range"]

            - nominal["range"]
        )


        # =================================
        # Residuals
        # =================================

        loss_temp_plus = (

            temp_plus_model

            - case["temp_plus"]
        )**2


        loss_temp_minus = (

            temp_minus_model

            - case["temp_minus"]
        )**2

        # =================================
        # Ratio Constraint
        # =================================

        if abs(temp_plus_model) > 1e-6:

            ratio_model = (

                abs(temp_minus_model)

                /

                abs(temp_plus_model)
            )

            loss_ratio = (

                ratio_model

                - 3.0
            )**2

        else:

            loss_ratio = 1000.0


        # =================================
        # Add To Total Loss
        # =================================

        total_loss += (

            loss_temp_plus

            +

            loss_temp_minus

            +

            5.0 * loss_ratio
        )


        # =================================
        # Debug Print
        # =================================

        print("\n----------------------------------------")
        print(case["name"])
        print("----------------------------------------")

        print()

        print(

            f"Observed Warm : "

            f"{case['temp_plus']:.3f}"
        )

        print(

            f"Model Warm    : "

            f"{temp_plus_model:.3f}"
        )

        print()

        print(

            f"Observed Cold : "

            f"{case['temp_minus']:.3f}"
        )

        print(

            f"Model Cold    : "

            f"{temp_minus_model:.3f}"
        )

        print()


    # =====================================
    # Iteration Print
    # =====================================

    print("\n========================================")
    print("ITERATION")
    print("========================================\n")

    print(f"LOSS : {total_loss:.6f}")

    print()

    print(f"A  : {params['A']:.6f}")

    print(f"M1 : {params['M1']:.6f}")

    print(f"k1 : {params['k1']:.6f}")

    print()

    return total_loss


# =========================================
# Initial Guess
# =========================================

x0 = np.array([

    1.08,

    0.948035,

    350
])


# =========================================
# Bounds
# =========================================

bounds = [

    (0.1, 1.5),      # A

    (0.93, 0.97),    # M1

    (10.0, 1000)   # k1
]


# =========================================
# Optimize
# =========================================

result = minimize(

    objective,

    x0,

    method="L-BFGS-B",

    bounds=bounds,

    options={

        "maxiter": 20
    }
)


# =========================================
# Final Result
# =========================================

print("\n========================================")
print("OPTIMIZATION RESULT")
print("========================================\n")

print(result)

print()

print("Optimized Parameters:\n")

print(f"A  = {result.x[0]:.6f}")

print(f"M1 = {result.x[1]:.6f}")

print(f"k1 = {result.x[3]:.6f}")
