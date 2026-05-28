# optimize2.py

import numpy as np

from scipy.optimize import minimize

from main import run_case

from cases import cases


# =========================================
# Fixed Hump Parameters
# =========================================

FIXED = {

    "A": 0.552486,

    "M1": 0.942764,

    "M2": 1.085635,

    "k1": 149.999543,

    "k2": 7.021692
}


# =========================================
# Objective Function
# =========================================

def objective(x):

    # =====================================
    # Parameters To Optimize
    # =====================================

    params = {

        "C0": x[0],

        "C1": x[1],

        "C2": x[2],

        "A": FIXED["A"],

        "M1": FIXED["M1"],

        "M2": FIXED["M2"],

        "k1": FIXED["k1"],

        "k2": FIXED["k2"]
    }


    # =====================================
    # Total Loss
    # =====================================

    total_loss = 0.0


    # =====================================
    # Loop Over Cases
    # =====================================

    for case in cases:


        # ---------------------------------
        # Skip Disabled Cases
        # ---------------------------------

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
        # Density +1%
        # =================================

        density_plus = run_case(

            v0=case["v0"],

            theta_mil=case["theta_mil"],

            density_scale=1.01,

            temp_scale=1.0,

            params=params
        )


        # =================================
        # Density -1%
        # =================================

        density_minus = run_case(

            v0=case["v0"],

            theta_mil=case["theta_mil"],

            density_scale=0.99,

            temp_scale=1.0,

            params=params
        )


        # =================================
        # Model Corrections
        # =================================

        density_plus_model = (

            density_plus["range"]

            - nominal["range"]
        )

        density_minus_model = (

            density_minus["range"]

            - nominal["range"]
        )


        # =================================
        # Nominal Residuals
        # =====================================

        loss_range = (

            nominal["range"]

            - case["range"]
        )**2


        loss_tof = (

            nominal["tof"]

            - case["tof"]
        )**2


        loss_hmax = (

            nominal["hmax"]

            - case["hmax"]
        )**2


        loss_impact_velocity = (

            nominal["impact_velocity"]

            - case["impact_velocity"]
        )**2


        loss_impact_angle = (

            nominal["impact_angle"]

            - case["impact_angle"]
        )**2


        # =================================
        # Density Residuals
        # =====================================

        loss_density_plus = (

            density_plus_model

            - case["density_plus"]
        )**2


        loss_density_minus = (

            density_minus_model

            - case["density_minus"]
        )**2


        # =================================
        # Weighted Loss
        # =====================================

        case_loss = (

            1.0 * loss_range

            +

            1.0 * loss_tof

            +

            0.01 * loss_hmax

            +

            0.5 * loss_impact_velocity

            +

            0.5 * loss_impact_angle

            +

            2.0 * loss_density_plus

            +

            2.0 * loss_density_minus
        )


        total_loss += case_loss


        # =================================
        # Debug Print
        # =====================================

        print("\n----------------------------------------")
        print(case["name"])
        print("----------------------------------------")

        print()

        print(

            f"Range Obs   : {case['range']:.2f}"
        )

        print(

            f"Range Model : {nominal['range']:.2f}"
        )

        print()

        print(

            f"Density+ Obs   : "

            f"{case['density_plus']:.2f}"
        )

        print(

            f"Density+ Model : "

            f"{density_plus_model:.2f}"
        )

        print()

        print(

            f"Density- Obs   : "

            f"{case['density_minus']:.2f}"
        )

        print(

            f"Density- Model : "

            f"{density_minus_model:.2f}"
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

    print(f"C0 : {params['C0']:.6f}")

    print(f"C1 : {params['C1']:.6f}")

    print(f"C2 : {params['C2']:.6f}")

    print()

    return total_loss


# =========================================
# Initial Guess
# =========================================

x0 = np.array([

    0.142,

    -0.01,

    0.0
])


# =========================================
# Bounds
# =========================================

bounds = [

    (0.05, 0.30),      # C0

    (-0.10, 0.05),      # C1

    (-0.10, 0.10)      # C2
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

        "maxiter": 200
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

print(f"C0 = {result.x[0]:.6f}")

print(f"C1 = {result.x[1]:.6f}")

print(f"C2 = {result.x[2]:.6f}")