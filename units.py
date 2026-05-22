import numpy as np


# =========================
# Angle Conversions
# =========================

def mil_to_deg(mil):

    return mil * (360.0 / 6400.0)


def deg_to_mil(deg):

    return deg * (6400.0 / 360.0)


def mil_to_rad(mil):

    deg = mil_to_deg(mil)

    return np.radians(deg)


def rad_to_mil(rad):

    deg = np.degrees(rad)

    return deg_to_mil(deg)