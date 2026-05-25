import numpy as np

from aero import (

    mach_table,

    base_cd_table
)


# =========================================
# Transonic Hump
# =========================================

def hump(M):

    return np.exp(

        -((M - 0.82) / 0.06)**2
    )


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