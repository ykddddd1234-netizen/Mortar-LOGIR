import numpy as np


# =========================================
# MET Layer Midpoints
# =========================================

z_nodes = np.array([

    100,    # 00
    350,    # 01
    750,    # 02
    1250,   # 03
    1750,   # 04
    2250,   # 05
    2750    # 06
])


# =========================================
# Default MET Ratios
# =========================================

default_density_ratios = np.ones(7)

default_temp_ratios = np.ones(7)


# =========================================
# Active MET Ratios
# =========================================

density_ratios = default_density_ratios.copy()

temp_ratios = default_temp_ratios.copy()


# =========================================
# Set Density Ratios
# =========================================

def set_density_ratios(values):

    global density_ratios

    density_ratios[:] = values


# =========================================
# Set Temperature Ratios
# =========================================

def set_temp_ratios(values):

    global temp_ratios

    temp_ratios[:] = values


# =========================================
# Uniform Density Perturbation
# =========================================

def set_uniform_density_ratio(value):

    global density_ratios

    density_ratios[:] = value


# =========================================
# Uniform Temperature Perturbation
# =========================================

def set_uniform_temp_ratio(value):

    global temp_ratios

    temp_ratios[:] = value


# =========================================
# Reset MET
# =========================================

def reset_met():

    global density_ratios

    global temp_ratios


    density_ratios[:] = (

        default_density_ratios
    )

    temp_ratios[:] = (

        default_temp_ratios
    )


# =========================================
# Interpolated MET Ratios
# =========================================

def get_met_ratios(z):


    # =====================================
    # Above Line 06
    # =====================================

    if z >= z_nodes[-1]:

        return 1.0, 1.0


    # =====================================
    # Density Ratio
    # =====================================

    density_ratio = np.interp(

        z,

        z_nodes,

        density_ratios
    )


    # =====================================
    # Temperature Ratio
    # =====================================

    temp_ratio = np.interp(

        z,

        z_nodes,

        temp_ratios
    )


    return density_ratio, temp_ratio