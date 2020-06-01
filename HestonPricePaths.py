import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform, randn

def heston(S0, mu, v0, rho, kappa, theta, xi, T, timesteps):

    # Generate random Brownian Motion
    MU  = np.array([0, 0])
    COV = np.matrix([[1, rho], [rho, 1]]) 
    W   = np.random.multivariate_normal(MU, COV, T) #T 2-d vector samples from distribution
    eps_S = W[:,0] #dW is modelled as sqrt(dt)*eps where eps is sample from distrubution
    eps_v = W[:,1]

    # Generate paths
    vt    = np.zeros(T)
    vt[0] = v0
    St    = np.zeros(T)
    St[0] = S0

    dt = T/timesteps

    for t in range(1,timesteps):
        vt[t] = np.abs(vt[t-1] + kappa*(theta-vt[t-1])*dt + xi*np.sqrt(vt[t-1]*dt)*eps_v[t])
        St[t] = St[t-1]*np.exp((mu - 0.5*vt[t-1])*dt + np.sqrt(vt[t-1]*dt)*eps_S[t])

    return St, vt


T     = 365
timesteps = 365
S0    = 1 # Initial price
mu    = 1 # Expected return
sigma = 0.2 # Volatility
rho   = -0.2 # Correlation
kappa = 0.5 # Revert rate
theta = 0.9 # Long-term volatility
xi    = 0.5 # Volatility of instantaneous volatility
v0    = 0.9 # Initial instantaneous volatility


hestpath = heston(S0, mu, v0, rho, kappa, theta, xi, T, timesteps)
stock = hestpath[0]
volatility = hestpath[1]

plt.plot(stock)
plt.show()
plt.plot(volatility)
plt.show()