# optimize.py

import numpy as np

from scipy.optimize import minimize

from simulate import simulate

from cases import cases

from met import (

    reset_met,

    set_uniform_density_ratio,

    set_uniform_temp_ratio
)

# Initial Setting

last_loss = None

# =========================================================
# Optimization Flags
# =========================================================

USE_DENSITY = False

USE_TEMPERATURE = True


# =========================================================
# Parameter Conversion
# =========================================================

def vector_to_params(x):

    return {

        "C0": x[0],

        "C1": x[1],

        "A": x[2],

        "Mc": x[3],

        "k": x[4],

        "tau": x[5]

    }


# =========================================================
# Nominal Error
# =========================================================

def compute_nominal_error(

    result,

    case
):

    e_range = (

        (
            result["range"]
            - case["range"]
        )

        / case["range"]
    )


    e_tof = (

        (
            result["tof"]
            - case["tof"]
        )

        / case["tof"]
    )


    e_hmax = (

        (
            result["hmax"]
            - case["hmax"]
        )

        / case["hmax"]
    )


    e_v = (

        (
            result["impact_velocity"]
            - case["impact_velocity"]
        )

        / case["impact_velocity"]
    )


    e_angle = (

        (
            result["impact_angle"]
            - case["impact_angle"]
        )

        / case["impact_angle"]
    )


    return (

        15.0 * e_range**2 +

        2.0 * e_tof**2 +

        5.0 * e_hmax**2 +

        1.0 * e_v**2 +

        1.0 * e_angle**2
    )


# =========================================================
# Correction Error
# =========================================================

def compute_correction_error(

    delta_sim,

    delta_target
):

    scale = max(

        abs(delta_target),

        1.0
    )

    return (

        (
            delta_sim
            - delta_target
        )

        / scale
    )**2


# =========================================================
# Objective Function
# =========================================================

def objective(x):

    params = vector_to_params(x)

    total_error = 0.0


    # =====================================================
    # Loop Cases
    # =====================================================

    for case in cases:


        # =================================================
        # Nominal Atmosphere
        # =================================================

        reset_met()

        nominal = simulate(

            case["v0"],

            case["theta_mil"],

            params
        )


        # =================================================
        # Nominal Error
        # =================================================

        total_error += compute_nominal_error(

            nominal,

            case
        )


        # =================================================
        # Density Corrections
        # =================================================

        if USE_DENSITY:


            # =============================================
            # Density +1%
            # =============================================

            set_uniform_density_ratio(

                1.01
            )

            rho_plus = simulate(

                case["v0"],

                case["theta_mil"],

                params
            )


            # =============================================
            # Density -1%
            # =============================================

            set_uniform_density_ratio(

                0.99
            )

            rho_minus = simulate(

                case["v0"],

                case["theta_mil"],

                params
            )


            # =============================================
            # Reset
            # =============================================

            reset_met()


            # =============================================
            # Corrections
            # =============================================

            delta_rho_plus = (

                rho_plus["range"]

                - nominal["range"]
            )

            delta_rho_minus = (

                rho_minus["range"]

                - nominal["range"]
            )


            # =============================================
            # Error
            # =============================================

            total_error += (

                0.5 *

                compute_correction_error(

                    delta_rho_plus,

                    case["density_plus"]
                )
            )


            total_error += (

                0.5 *

                compute_correction_error(

                    delta_rho_minus,

                    case["density_minus"]
                )
            )


        # =================================================
        # Temperature Corrections
        # =================================================

        if USE_TEMPERATURE:


            # =============================================
            # Temp +1%
            # =============================================

            set_uniform_temp_ratio(

                1.01
            )

            temp_plus = simulate(

                case["v0"],

                case["theta_mil"],

                params
            )


            # =============================================
            # Temp -1%
            # =============================================

            set_uniform_temp_ratio(

                0.99
            )

            temp_minus = simulate(

                case["v0"],

                case["theta_mil"],

                params
            )


            # =============================================
            # Reset
            # =============================================

            reset_met()


            # =============================================
            # Corrections
            # =============================================

            delta_temp_plus = (

                temp_plus["range"]

                - nominal["range"]
            )

            delta_temp_minus = (

                temp_minus["range"]

                - nominal["range"]
            )


            # =============================================
            # Error
            # =============================================

            total_error += (

                1.0 *

                compute_correction_error(

                    delta_temp_plus,

                    case["temp_plus"]
                )
            )


            total_error += (

                1.0 *

                compute_correction_error(

                    delta_temp_minus,

                    case["temp_minus"]
                )
            )

    global last_loss

    last_loss = total_error

    return total_error


# =========================================================
# Callback
# =========================================================

iteration = 0


def callback(xk):

    global iteration

    global last_loss


    iteration += 1

    params = vector_to_params(xk)


    print("\n================================================")
    print(f"ITERATION : {iteration}")
    print("================================================")


    print(

        f"\nLOSS : {last_loss:.8f}\n"
    )


    for key, value in params.items():

        print(

            f"{key:>4} : {value:.6f}"
        )


# =========================================================
# Initial Guess
# =========================================================

x0 = np.array([

    0.14,       # C0

    -0.02,      # C1

    0.15,       # A

    0.95,       # Mc

    30,         # k

    1.0         # tau
])


# =========================================================
# Bounds
# =========================================================

bounds = [

    (0.08, 0.20),     # C0

    (-0.05, 0.05),    # C1

    (0.00, 0.30),     # A

    (0.80, 1.00),     # Mc

    (1.0, 100),      # k

    (0.01, 10.0)       # tau
]


# =========================================================
# Optimization
# =========================================================

result = minimize(

    objective,

    x0,

    method="L-BFGS-B",

    bounds=bounds,

    callback=callback,

    options={

        "maxiter": 25
    }
)


# =========================================================
# Best Parameters
# =========================================================

best_params = vector_to_params(

    result.x
)


# =========================================================
# Summary
# =========================================================

print("\n================================================")
print("OPTIMIZATION COMPLETE")
print("================================================\n")

print(

    "SUCCESS :",

    result.success
)

print(

    "MESSAGE :",

    result.message
)

print(

    "ITERATIONS :",

    result.nit
)

print(

    "FUNCTION EVALS :",

    result.nfev
)

print(

    f"\nFINAL LOSS : {result.fun:.8f}"
)


# =========================================================
# Best Parameters
# =========================================================

print("\n================================================")
print("BEST PARAMETERS")
print("================================================\n")


for key, value in best_params.items():

    print(

        f"{key:>4} : {value:.6f}"
    )


# =========================================================
# Final Results
# =========================================================

print("\n================================================")
print("FINAL CASE RESULTS")
print("================================================")


for case in cases:


    # =====================================================
    # Nominal
    # =====================================================

    reset_met()

    nominal = simulate(

        case["v0"],

        case["theta_mil"],

        best_params
    )


    print("\n------------------------------------------------")
    print(case["name"])
    print("------------------------------------------------")


    # =====================================================
    # Nominal
    # =====================================================

    print(

        f"\nRange : "

        f"{nominal['range']:.2f} "

        f"(Target {case['range']:.2f})"
    )


    print(

        f"TOF : "

        f"{nominal['tof']:.2f} "

        f"(Target {case['tof']:.2f})"
    )


    print(

        f"HMAX : "

        f"{nominal['hmax']:.2f} "

        f"(Target {case['hmax']:.2f})"
    )


    print(

        f"Impact Velocity : "

        f"{nominal['impact_velocity']:.2f} "

        f"(Target {case['impact_velocity']:.2f})"
    )


    print(

        f"Impact Angle : "

        f"{nominal['impact_angle']:.2f} "

        f"(Target {case['impact_angle']:.2f})"
    )


    # =====================================================
    # Density Corrections
    # =====================================================

    if USE_DENSITY:


        set_uniform_density_ratio(

            1.01
        )

        rho_plus = simulate(

            case["v0"],

            case["theta_mil"],

            best_params
        )


        set_uniform_density_ratio(

            0.99
        )

        rho_minus = simulate(

            case["v0"],

            case["theta_mil"],

            best_params
        )


        reset_met()


        delta_rho_plus = (

            rho_plus["range"]

            - nominal["range"]
        )


        delta_rho_minus = (

            rho_minus["range"]

            - nominal["range"]
        )


        print(

            f"\nDensity +1% : "

            f"{delta_rho_plus:.2f} "

            f"(Target {case['density_plus']:.2f})"
        )


        print(

            f"Density -1% : "

            f"{delta_rho_minus:.2f} "

            f"(Target {case['density_minus']:.2f})"
        )


    # =====================================================
    # Temperature Corrections
    # =====================================================

    if USE_TEMPERATURE:


        set_uniform_temp_ratio(

            1.01
        )

        temp_plus = simulate(

            case["v0"],

            case["theta_mil"],

            best_params
        )


        set_uniform_temp_ratio(

            0.99
        )

        temp_minus = simulate(

            case["v0"],

            case["theta_mil"],

            best_params
        )


        reset_met()


        delta_temp_plus = (

            temp_plus["range"]

            - nominal["range"]
        )


        delta_temp_minus = (

            temp_minus["range"]

            - nominal["range"]
        )


        print(

            f"\nTemp +1% : "

            f"{delta_temp_plus:.2f} "

            f"(Target {case['temp_plus']:.2f})"
        )


        print(

            f"Temp -1% : "

            f"{delta_temp_minus:.2f} "

            f"(Target {case['temp_minus']:.2f})"
        )