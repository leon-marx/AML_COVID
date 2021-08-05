import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


def create_toydata(T, I0, R0, N, nu, beta, gamma, mu):
    """
    Returns Toy Data for T days given initial conditions I0, R0
    and pandemic-specific parameters nu, beta, gamma, mu:
        T: Number of days
        I0: Initial number of infected people
        R0: Initial number of recovered people
        N: Number of individuals
        nu: Birth rate per person
        beta: Number of new infections for one first infected person per day
        gamma: Death / Recovery rate per day
        mu: General death rate per person and day
    Data is returned as a numpy array of shape (T, 3), with [S, I, R] triplets for each timestep.
    """

    def SIR_ode(y, t, N, nu, beta, gamma, mu):
        S, I, R = y

        S_deriv = nu * N - beta * S * I / N - mu * S
        I_deriv = beta * S * I / N - gamma * I - mu * I
        R_deriv = gamma * I - mu * R

        y_deriv = S_deriv, I_deriv, R_deriv
        return y_deriv

    S0 = N - I0 - R0
    y0 = S0, I0, R0
    t = np.arange(T)
    args = (N, nu, beta, gamma, mu)

    result = odeint(SIR_ode, y0, t, args=args)
        
    return result


"""
Example usage:

T = 10
I0 = 1
R0 = 0
N = 100
nu = 0.001
beta = 1.3
gamma = 0.3
mu = 0.0001

Timeseries = create_toydata(T, I0, R0, N, nu, beta, gamma, mu)
print(Timeseries)
"""
