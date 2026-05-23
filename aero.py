import numpy as np

from scipy.interpolate import PchipInterpolator


# =========================================
# Aero Table
# =========================================

mach_table = np.array([

    0.0,
    0.3,
    0.5,
    0.7,

    0.74,
    0.76,
    0.78,
    0.80,
    0.82,
    0.84,
    0.86,
    0.88,
    0.90,

    1.0,
    1.1,
    1.2
])


# =========================================
# Cd(M)
# =========================================

def get_cd(M, cd_table):

    interp = PchipInterpolator(

        mach_table,

        cd_table
    )

    return float(interp(M))