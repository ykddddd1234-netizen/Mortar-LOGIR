#rk4.py
def rk4_step(state, dt, derivatives):

    k1 = derivatives(state)

    k2 = derivatives(
        state + 0.5 * dt * k1
    )

    k3 = derivatives(
        state + 0.5 * dt * k2
    )

    k4 = derivatives(
        state + dt * k3
    )

    return (
        state
        + dt/6.0
        * (k1 + 2*k2 + 2*k3 + k4)
    )