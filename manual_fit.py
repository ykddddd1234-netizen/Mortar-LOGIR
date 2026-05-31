#manual_fit.py
from main import run_case
from cases import cases


# manual_fit.py 에서 사용할 새로운 params 예시
params = {
    # base_mach_nodes 8개 축과 1:1 매칭되는 항력 값
    "baseline_cds": [
        0.1340,  # Node 0 (M=0.00) : 고정
        0.1278,  # Node 1 (M=0.65) : 고정
        0.1270,  # Node 2 (M=0.71) : 고정
        0.1265,  # Node 3 (M=0.77) : 고정
        0.1340,  # Node 4 (M=0.805): 고정
        0.1415,  # Node 5 (M=0.83)  : 고도 방어선 완벽하므로 0.1415 유지 (고도 고정)
        0.1545,  # ★ 4000m 대역(+12.6m)을 10m 안으로 밀어 넣기 위해 상향 (0.1485 -> 0.1545)
        0.1800,  # ★ 5300m 대역(+18.4m)의 사거리 브레이크 쐐기 수치 (0.1650 -> 0.1800)
        0.1350,  # Node 8 (M=0.89)  : 고정 (장약 6 영점선)
        0.1323   # Node 9 (M=0.94)  : 고정
    ], 
    
    # 천음속 절벽 형상 파라미터 (절대 고정)
    "A": 1.079934,
    "M1": 0.947904,
    "M2": 1.1,
    "k1": 350.000012,
    "k2": 7.0
}


print("\n========================================")
print("MANUAL FIT")
print("========================================")


for case in cases:

    if not case.get("active", True):
        continue

    nominal = run_case(

        v0=case["v0"],

        theta_mil=case["theta_mil"],

        density_scale=1.0,

        temp_scale=1.0,

        params=params
    )

    warm = run_case(

        v0=case["v0"],

        theta_mil=case["theta_mil"],

        density_scale=1.0,

        temp_scale=1.01,

        params=params
    )

    cold = run_case(

        v0=case["v0"],

        theta_mil=case["theta_mil"],

        density_scale=1.0,

        temp_scale=0.99,

        params=params
    )

    density_plus = run_case(

        v0=case["v0"],

        theta_mil=case["theta_mil"],

        density_scale=1.01,

        temp_scale=1.0,

        params=params
    )

    density_minus = run_case(

        v0=case["v0"],

        theta_mil=case["theta_mil"],

        density_scale=0.99,

        temp_scale=1.0,

        params=params
    )

    temp_plus_model = (

        warm["range"]

        - nominal["range"]
    )

    temp_minus_model = (

        cold["range"]

        - nominal["range"]
    )

    density_plus_model = (

        density_plus["range"]

        - nominal["range"]
    )

    density_minus_model = (

        density_minus["range"]

        - nominal["range"]
    )

    print("\n----------------------------------------")
    print(case["name"])
    print("----------------------------------------")

    print()

    print(
        f"Range    : "
        f"{nominal['range']:.2f}"
        f" / "
        f"{case['range']:.2f}"
    )

    print(
        f"TOF      : "
        f"{nominal['tof']:.2f}"
        f" / "
        f"{case['tof']:.2f}"
    )

    print(
        f"HMAX     : "
        f"{nominal['hmax']:.2f}"
        f" / "
        f"{case['hmax']:.2f}"
    )

    print(
        f"Vimpact  : "
        f"{nominal['impact_velocity']:.2f}"
        f" / "
        f"{case['impact_velocity']:.2f}"
    )

    print(
        f"Angle    : "
        f"{nominal['impact_angle']:.2f}"
        f" / "
        f"{case['impact_angle']:.2f}"
    )

    print()

    print(
        f"Temp+    : "
        f"{temp_plus_model:.2f}"
        f" / "
        f"{case['temp_plus']:.2f}"
    )

    print(
        f"Temp-    : "
        f"{temp_minus_model:.2f}"
        f" / "
        f"{case['temp_minus']:.2f}"
    )

    print()

    print(
        f"Density+ : "
        f"{density_plus_model:.2f}"
        f" / "
        f"{case['density_plus']:.2f}"
    )

    print(
        f"Density- : "
        f"{density_minus_model:.2f}"
        f" / "
        f"{case['density_minus']:.2f}"
    )


import numpy as np
import matplotlib.pyplot as plt

from aero import get_cd


# =====================================
# Mach Axis
# =====================================

mach = np.linspace(

    0.5,

    1.10,

    1000
)

# =====================================
# Cd Curves
# =====================================

cd_nominal = [

    get_cd(

        M,

        params
    )

    for M in mach
]

# =====================================
# 320 m/s Mach
# =====================================

a0 = 340.0

M320_nominal = 320.0 / a0

M320_warm = (

    320.0

    /

    (a0 * np.sqrt(1.01))
)

M320_cold = (

    320.0

    /

    (a0 * np.sqrt(0.99))
)


print()

print("320 m/s Mach")

print(

    f"Warm    : {M320_warm:.4f}"
)

print(

    f"Nominal : {M320_nominal:.4f}"
)

print(

    f"Cold    : {M320_cold:.4f}"
)

print()


# =====================================
# Plot
# =====================================

plt.figure(

    figsize=(10, 6)
)

plt.plot(

    mach,

    cd_nominal,

    linewidth=3,

    label="Nominal"
)

# -------------------------------------
# 320 m/s Markers
# -------------------------------------

plt.axvline(

    M320_nominal,

    linestyle="-",

    linewidth=2,

    alpha=0.8,

    label=f"320m/s Nom ({M320_nominal:.3f})"
)

plt.axvline(

    M320_warm,

    linestyle="--",

    linewidth=2,

    alpha=0.8,

    label=f"320m/s Warm ({M320_warm:.3f})"
)

plt.axvline(

    M320_cold,

    linestyle="--",

    linewidth=2,

    alpha=0.8,

    label=f"320m/s Cold ({M320_cold:.3f})"
)


plt.xlabel("Mach")

plt.ylabel("Cd")

plt.title("Cd(Mach)")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(

    "cd_temp_shift.png",

    dpi=300,

    bbox_inches="tight"
)

print()

print("Saved : cd_temp_shift.png")

print()