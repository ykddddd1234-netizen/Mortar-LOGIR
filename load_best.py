import numpy as np

from cd_builder import build_charge_cd_table

from aero import mach_table

import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator


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
# Build Tables
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
# Print
# =========================================

print("A3 :", A3)

print("A4 :", A4)

print("A5 :", A5)

print("A6 :", A6)

print()

print("Charge 3")

print(cd_table_3)

print()

print("Charge 4")

print(cd_table_4)

print()

print("Charge 5")

print(cd_table_5)

print()

print("Charge 6")

print(cd_table_6)


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

plt.legend()

plt.grid(True)

plt.show()