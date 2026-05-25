import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator

from simulate import simulate
from cases import cases

from aero import (
    mach_table,
    base_cd_table
)

iteration = 0

last_error = None

best_error = np.inf

best_params = None


# =========================================
# Transonic Hump
# =========================================

def hump(M):

    return np.exp(
        -((M - 0.82) / 0.06)**2
    )


# =========================================
# Build Cd Table
# =========================================

def build_cd_table(

    global_delta,

    charge_scales
):

    cd_table = np.zeros_like(base_cd_table)

    for i, M in enumerate(mach_table):

        cd_table[i] = (

            base_cd_table[i]

            +

            global_delta[i]

            +

            hump(M)
        )

    return cd_table


# =========================================
# Build Charge Cd Table
# =========================================

def build_charge_cd_table(

    global_delta,

    charge_scale
):

    cd_table = np.zeros_like(base_cd_table)

    for i, M in enumerate(mach_table):

        cd_table[i] = (

            base_cd_table[i]

            +

            global_delta[i]

            +

            charge_scale * hump(M)
        )

    return cd_table


# =========================================
# Objective Function
# =========================================

def objective(params):

    # =====================================
    # Parameter Split
    # =====================================

    global_delta = params[:16]

    A3 = params[16]

    A4 = params[17]

    A5 = params[18]

    A6 = params[19]


    # =====================================
    # Charge Scales
    # =====================================

    charge_scale_map = {

        "charge_3": A3,

        "charge_4": A4,

        "charge_5": A5,

        "charge_6": A6
    }


    total_error = 0.0

    smoothness = 0.0


    # =====================================
    # Smoothness Penalty
    # =====================================

    for i in range(len(global_delta) - 1):

        smoothness += (

            global_delta[i+1]

            -

            global_delta[i]

        )**2


    # =====================================
    # Case Loop
    # =====================================

    for case in cases:

        charge_name = case["name"]

        charge_scale = charge_scale_map[charge_name]


        cd_table = build_charge_cd_table(

            global_delta,

            charge_scale
        )


        result = simulate(

            v0=case["v0"],

            theta_mil=case["theta_mil"],

            cd_table=cd_table
        )


        # =================================
        # Relative Errors
        # =================================

        range_error = (

            result["range"]

            -

            case["range"]

        ) / case["range"]


        tof_error = (

            result["tof"]

            -

            case["tof"]

        ) / case["tof"]


        angle_error = (

            result["impact_angle"]

            -

            case["impact_angle"]

        ) / case["impact_angle"]


        velocity_error = (

            result["impact_velocity"]

            -

            case["impact_velocity"]

        ) / case["impact_velocity"]


        hmax_error = (

            result["hmax"]

            -

            case["hmax"]

        ) / case["hmax"]


        # =================================
        # Weighted Error
        # =================================

        total_error += (

            5.0 * range_error**2

            +

            2.0 * tof_error**2

            +

            2.0 * angle_error**2

            +

            1.0 * velocity_error**2

            +

            1.0 * hmax_error**2
        )


    # =====================================
    # Regularization
    # =====================================

    lambda_smooth = 1

    fit_error = total_error


    regularization = (

        lambda_smooth

        *

        smoothness
    )


    total_error += regularization


    print(

        "Fit Error :",

        fit_error
    )

    print(

        "Regularization :",

        regularization
    )


    global last_error

    last_error = total_error


    return total_error


# =========================================
# Initial Guess
# =========================================

global_delta0 = np.zeros(16)

A3_0 = 0.000

A4_0 = 0.000

A5_0 = 0.000

A6_0 = 0.000


initial_guess = np.concatenate([

    global_delta0,

    [

        A3_0,

        A4_0,

        A5_0,

        A6_0
    ]
])


# =========================================
# Bounds
# =========================================

bounds = []


# Global Delta Bounds

for _ in range(16):

    bounds.append(

        (-0.02, 0.02)
    )


# Charge Scale Bounds

bounds.extend([

    (-0.05, 0.05),  # A3

    (-0.05, 0.05),  # A4

    (-0.05, 0.05),  # A5

    (-0.05, 0.05)   # A6
])

# =========================================
# callback
# =========================================

def callback(xk):

    global iteration

    global best_error

    global best_params

    iteration += 1


    if last_error < best_error:

        best_error = last_error

        best_params = xk.copy()

        np.save(

            "best_params.npy",

            best_params
        )


    print(

        f"Iteration : {iteration}"
    )

    print(

        f"Current Error : {last_error}"
    )

    print()

# =========================================
# Optimization
# =========================================

result = minimize(

    objective,

    initial_guess,

    method="L-BFGS-B",

    bounds=bounds,

    callback=callback,

    options={

        "maxiter": 10
    }
)


# =========================================
# Optimized Parameters
# =========================================

global_delta_opt = result.x[:16]

A3_opt = result.x[16]

A4_opt = result.x[17]

A5_opt = result.x[18]

A6_opt = result.x[19]


# =========================================
# Final Tables
# =========================================

cd_table_3 = build_charge_cd_table(

    global_delta_opt,

    A3_opt
)

cd_table_4 = build_charge_cd_table(

    global_delta_opt,

    A4_opt
)

cd_table_5 = build_charge_cd_table(

    global_delta_opt,

    A5_opt
)

cd_table_6 = build_charge_cd_table(

    global_delta_opt,

    A6_opt
)


# =========================================
# Results
# =========================================

print("\n===== Optimization Result =====\n")

print("A3 :", A3_opt)

print("A4 :", A4_opt)

print("A5 :", A5_opt)

print("A6 :", A6_opt)

print()

print("Total Error :", result.fun)

print()


# =========================================
# Validation
# =========================================

for case in cases:

    charge_name = case["name"]


    if charge_name == "charge_3":

        cd_table = cd_table_3

    elif charge_name == "charge_4":

        cd_table = cd_table_4

    elif charge_name == "charge_5":

        cd_table = cd_table_5

    else:

        cd_table = cd_table_6


    sim = simulate(

        v0=case["v0"],

        theta_mil=case["theta_mil"],

        cd_table=cd_table
    )


    print("====================================")

    print(charge_name)

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
# Plot
# =========================================

interp3 = PchipInterpolator(

    mach_table,

    cd_table_3
)

interp4 = PchipInterpolator(

    mach_table,

    cd_table_4
)

interp5 = PchipInterpolator(

    mach_table,

    cd_table_5
)

interp6 = PchipInterpolator(

    mach_table,

    cd_table_6
)


M_plot = np.linspace(

    0.0,

    1.2,

    500
)


plt.figure(figsize=(10,6))

plt.plot(

    M_plot,

    interp3(M_plot),

    label="Charge 3"
)

plt.plot(

    M_plot,

    interp4(M_plot),

    label="Charge 4"
)

plt.plot(

    M_plot,

    interp5(M_plot),

    label="Charge 5"
)

plt.plot(

    M_plot,

    interp6(M_plot),

    label="Charge 6"
)

plt.scatter(

    mach_table,

    base_cd_table,

    color='black',

    zorder=5,

    label="Base Table"
)

plt.xlabel("Mach Number")

plt.ylabel("Cd")

plt.title("Optimized Charge-dependent Cd(M)")

plt.grid(True)

plt.legend()

plt.savefig("optimized_cd_tables.png")

plt.show()