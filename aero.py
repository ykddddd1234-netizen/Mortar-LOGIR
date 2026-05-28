# aero.py

import numpy as np


# =========================================
# Baseline Cd
# =========================================

def baseline_cd(

    M,

    params
):

    C0 = params["C0"]

    C1 = params["C1"]

    C2 = params["C2"]


    return (

        C0

        +

        C1 * M

        +

        C2 * M**2
    )


# =========================================
# Hump Function
# =========================================

def hump(

    M,

    params
):

    A = params["A"]

    M1 = params["M1"]

    M2 = params["M2"]

    k1 = params["k1"]

    k2 = params["k2"]


    rise = 1.0 / (

        1.0

        + np.exp(

            -k1 * (M - M1)
        )
    )

    fall = 1.0 / (

        1.0

        + np.exp(

            k2 * (M - M2)
        )
    )


    return (

        A

        * rise

        * fall
    )


# =========================================
# Total Cd
# =========================================

def get_cd(

    M,

    params
):

    return (

        baseline_cd(

            M,

            params
        )

        +

        hump(

            M,

            params
        )
    )