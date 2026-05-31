# optimize2.py

import numpy as np

from scipy.optimize import minimize

from main import run_case

from cases import cases


# =========================================
# Fixed Hump Parameters
# =========================================

FIXED = {

    "A": 1.079934,

    "M1": 0.947904,

    "M2": 1.1,

    "k1": 350.000012,

    "k2": 7.0
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
        density_plus_model = density_plus["range"] - nominal["range"]
        density_minus_model = density_minus["range"] - nominal["range"]

        # =================================
        # Nominal Residuals (오차 계산)
        # =================================
        loss_range = (nominal["range"] - case["range"])**2
        loss_tof = (nominal["tof"] - case["tof"])**2
        loss_hmax = (nominal["hmax"] - case["hmax"])**2
        loss_impact_velocity = (nominal["impact_velocity"] - case["impact_velocity"])**2
        loss_impact_angle = (nominal["impact_angle"] - case["impact_angle"])**2

        # =================================
        # Density Residuals
        # =================================
        loss_density_plus = (density_plus_model - case["density_plus"])**2
        loss_density_minus = (density_minus_model - case["density_minus"])**2

        # =================================
        # Normalization Scales (사용자 정의 기준값의 제곱)
        # =================================
        # 각 물리량의 허용 오차(Tolerance)를 분모에 제곱으로 넣어 무차원화합니다.
        SCALE_RANGE = 10.0 ** 2              # 사거리 기준: 10m
        SCALE_TOF = 0.1 ** 2                # 비과시간 기준: 0.1초
        SCALE_HMAX = 10.0 ** 2              # 탄도고 기준: 10m
        SCALE_IMPACT_VELOCITY = 1.0 ** 2    # 종말속도 기준: 1m/s
        SCALE_IMPACT_ANGLE = 10.0 ** 2    # 낙각 기준: 10mil (약 0.5625도)
        SCALE_DENSITY = 1.0 ** 2            # 밀도 사거리 변화량 기준: 1m

        # =================================
        # Weighted & Normalized Loss
        # =================================
        # 모든 항목이 정규화되어 이제 오차가 기준치와 같으면 각각 정확히 1.0의 Loss를 가집니다.
        # 사거리에 조금 더 프리미엄을 주고 싶다면 앞의 가중치(1.0)를 2.0~3.0으로 올리면 됩니다.
        case_loss = (
            1.0 * (loss_range / SCALE_RANGE)
            + 1.0 * (loss_tof / SCALE_TOF)
            + 1.0 * (loss_hmax / SCALE_HMAX)
            + 1.0 * (loss_impact_velocity / SCALE_IMPACT_VELOCITY)
            + 1.0 * (loss_impact_angle / SCALE_IMPACT_ANGLE)
            + 1.0 * (loss_density_plus / SCALE_DENSITY)
            + 1.0 * (loss_density_minus / SCALE_DENSITY)
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

        range_residual = nominal["range"] - case["range"]

        tof_residual = nominal["tof"] - case["tof"]

        hmax_residual = nominal["hmax"] - case["hmax"]

        impact_velocity_residual = (
            nominal["impact_velocity"]
            - case["impact_velocity"]
        )

        impact_angle_residual = (
            nominal["impact_angle"]
            - case["impact_angle"]
        )

        print("----- Residuals -----")

        print(f"Range           : {range_residual:.3f}")

        print(f"TOF             : {tof_residual:.3f}")

        print(f"HMAX            : {hmax_residual:.3f}")

        print(
            f"Impact Velocity : "
            f"{impact_velocity_residual:.3f}"
        )

        print(
            f"Impact Angle    : "
            f"{impact_angle_residual:.3f}"
        )

        print()


        print()

        print(f"Case Loss       : {case_loss:.3f}")

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

    0.128794,

    -0.005527,

    0.010791
])


# =========================================
# Bounds
# =========================================

bounds = [

    (0.05, 0.30),      # C0

    (-0.10, 0.05),      # C1

    (-0.02, 0.02),      # C2
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