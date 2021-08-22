import numpy as np
from scipy.integrate import odeint


def create_toydata(T, I0, R0, N, beta, gamma):
    """
    Returns Toy Data for T days given initial conditions I0, R0
    and pandemic-specific parameters nu, beta, gamma, mu:
        T: Number of days
        I0: Initial number of infected people
        R0: Initial number of recovered people
        N: Number of individuals
        beta: Number of new infections for one first infected person per day
        gamma: Death / Recovery rate per day
    Data is returned as a numpy array of shape (T, 3), with [S, I, R] triplets for each timestep.
    """

    def SIR_ode(y, t, N, beta, gamma):
        S, I, R = y

        S_deriv = - beta * S * I / N
        I_deriv = beta * S * I / N - gamma * I
        R_deriv = gamma * I

        y_deriv = S_deriv, I_deriv, R_deriv
        return y_deriv

    S0 = N - I0 - R0
    y0 = S0, I0, R0
    t = np.arange(T)
    args = (N, beta, gamma)

    result = odeint(SIR_ode, y0, t, args=args)
        
    return result



# Example usage:
"""
T = 100
I0 = 5
R0 = 0
N = 100
beta = 0.4
gamma = 0.25

import matplotlib.pyplot as plt
Timeseries = create_toydata(T, I0, R0, N, beta, gamma).T
labels = ["Susceptible", "Infected", "Recovered"]
for i, col in enumerate(Timeseries):
    plt.plot(np.arange(len(col)), col, label=labels[i])
plt.legend()
plt.show()
"""
