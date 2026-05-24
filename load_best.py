# load_best.py

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator

from cd_builder import (

    build_charge_cd_table,

    hump
)

from aero import (

    mach_table,

    base_cd_table
)


# =========================================
# Load Best Parameters
# =========================================

params = np.load(

    "best_params.npy"
)


# =========================================
# Parameter Split
# =========================================

global_delta = params[:16]

A3 = params[16]

A4 = params[17]

A5 = params[18]

A6 = params[19]


# =========================================
# Optimized Base Cd
# =========================================

optimized_base_cd = (

    base_cd_table

    +

    global_delta
)


# =========================================
# Build Charge-dependent Cd Tables
# =========================================

cd_table_3 = build_charge_cd_table(

    global_delta,

    A3
)

cd_table_4 = build_charge_cd_table(

    global_delta,

    A4
)

cd_table_5 = build_charge_cd_table(

    global_delta,

    A5
)

cd_table_6 = build_charge_cd_table(

    global_delta,

    A6
)


# =========================================
# Print Parameters
# =========================================

print("\n===== Optimized Parameters =====\n")

print("A3 :", A3)

print("A4 :", A4)

print("A5 :", A5)

print("A6 :", A6)

print()


# =========================================
# Print Optimized Base Cd
# =========================================

print("===== Optimized Base Cd =====\n")

for M, Cd in zip(

    mach_table,

    optimized_base_cd
):

    print(

        f"Mach {M:.2f} : {Cd:.6f}"
    )

print()


# =========================================
# Save Tables as Python File
# =========================================

with open(

    "optimized_cd_tables.py",

    "w",

    encoding="utf-8"
) as f:


    # =====================================
    # Header
    # =====================================

    f.write(

        "# =========================================\n"
    )

    f.write(

        "# Optimized Charge-dependent Cd Tables\n"
    )

    f.write(

        "# =========================================\n\n"
    )


    # =====================================
    # Mach Table
    # =====================================

    f.write(

        "mach_table = "

        +

        repr([float(x) for x in mach_table])

        +

        "\n\n"
    )


    # =====================================
    # Optimized Base Cd
    # =====================================

    f.write(

        "optimized_base_cd = "

        +

        repr([float(x) for x in optimized_base_cd])

        +

        "\n\n"
    )


    # =====================================
    # Charge-dependent Cd Tables
    # =====================================

    f.write(

        "cd_table_3 = "

        +

        repr([float(x) for x in cd_table_3])

        +

        "\n\n"
    )


    f.write(

        "cd_table_4 = "

        +

        repr([float(x) for x in cd_table_4])

        +

        "\n\n"
    )


    f.write(

        "cd_table_5 = "

        +

        repr([float(x) for x in cd_table_5])

        +

        "\n\n"
    )


    f.write(

        "cd_table_6 = "

        +

        repr([float(x) for x in cd_table_6])

        +

        "\n\n"
    )


print(

    "Saved : optimized_cd_tables.py"
)


# =========================================
# Interpolators
# =========================================

interp_base = PchipInterpolator(

    mach_table,

    optimized_base_cd
)

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


# =========================================
# Plot Range
# =========================================

M_plot = np.linspace(

    0.0,

    1.2,

    500
)


# =========================================
# Plot
# =========================================

plt.figure(figsize=(10,6))


# =====================================
# Optimized Base
# =====================================

plt.plot(

    M_plot,

    interp_base(M_plot),

    linewidth=3,

    linestyle="--",

    label="Optimized Base Cd"
)


# =====================================
# Charge-dependent Curves
# =====================================

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


# =====================================
# Labels
# =====================================

plt.xlabel("Mach Number")

plt.ylabel("Cd")

plt.title("Optimized Charge-dependent Cd(M)")


# =====================================
# Grid / Legend
# =====================================

plt.grid(True)

plt.legend()


# =====================================
# Save Plot
# =====================================

plt.savefig(

    "cd_table.png",

    dpi=300
)

print(

    "Saved : cd_table.png"
)


# =====================================
# Close Figure
# =====================================

plt.close("all")