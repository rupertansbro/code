import numpy as np
import scipy.stats as ss
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt


class Heston_process():
    """
    Class for the Heston process:
    r = risk free constant rate
    rho = correlation between stock noise and variance noise
    theta = long term mean of the variance process
    sigma = volatility coefficient of the variance process
    kappa = mean reversion coefficient for the variance process
    """
    def __init__(self, mu=0.1, rho=0, sigma=0.2, theta=-0.1, kappa=0.1):
        self.mu = mu
        if (np.abs(rho)>1):
            raise ValueError("|rho| must be <=1")
        self.rho = rho
        if (theta<0 or sigma<0 or kappa<0):
            raise ValueError("sigma,theta,kappa must be positive")
        else:
            self.theta = theta
            self.sigma = sigma
            self.kappa = kappa            
    
    def path(self, S0, v0, N, T=1):
        """
        Produces one path of the Heston process.
        N = number of time steps
        T = Time in years
        Returns two arrays S (price) and v (variance). 
        """
        
        MU = np.array([0, 0])
        COV = np.matrix([[1, self.rho], [self.rho, 1]])
        W = ss.multivariate_normal.rvs( mean=MU, cov=COV, size=N-1 )
        W_S = W[:,0]                   # Stock Brownian motion:     W_1
        W_v = W[:,1]                   # Variance Brownian motion:  W_2

        # Initialize vectors
        T_vec, dt = np.linspace(0,T,N, retstep=True )
        dt_sq = np.sqrt(dt)
        
        X0 = np.log(S0)
        v = np.zeros(N)
        v[0] = v0
        X = np.zeros(N)
        X[0] = X0

        # Generate paths
        for t in range(0,N-1):
            v_sq = np.sqrt(v[t])
            v[t+1] = np.abs( v[t] + self.kappa*(self.theta - v[t])*dt + self.sigma * v_sq * dt_sq * W_v[t] )   
            X[t+1] = X[t] + (self.mu - 0.5*v[t])*dt + v_sq * dt_sq * W_S[t]
        return np.exp(X), v



N = 10                                                    # time points 
T = 10/365                                                    # time in years  
T_vec, dt = np.linspace(0,T,N, retstep=True )                 # time vector and time step
S0 = 100                                                      # initial price
v0 = 0.04                                                      # initial variance 
mu = 0.1; rho = -0.1; kappa = 5; theta = 0.04; sigma = 0.6    # alternative values: rho=-0.3 kappa=15 sigma=1
std_asy = np.sqrt( theta * sigma**2 /(2*kappa) )              # asymptotic standard deviation for the CIR process
assert(2*kappa * theta > sigma**2)                            # Feller condition



#particlefilter

np.random.seed(seed=42) 
Hest = Heston_process(mu=mu, rho=rho, sigma=sigma, theta=theta, kappa=kappa)
Strue, Vtrue = Hest.path(S0, v0, N, T)      # S is the stock, V is the variance

print('Strue = ', Strue)
print('Vtrue = ', Vtrue)

#fig = plt.figure(figsize=(16,4))
#ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122)
#ax1.plot(T_vec, Strue )
#ax1.set_title("Heston model, Stock process."); ax1.set_xlabel("Time"); ax1.set_ylabel("Stock")
#ax2.plot(T_vec, Vtrue )
#ax2.set_title("Heston model, Variance process."); ax2.set_xlabel("Time"); ax2.set_ylabel("Variance")
#ax2.plot(T_vec, (theta + std_asy)*np.ones_like(T_vec), label="1 asymptotic std dev", color="black" )
#ax2.plot(T_vec, (theta - std_asy)*np.ones_like(T_vec), color="black" )
#ax2.plot(T_vec, theta*np.ones_like(T_vec), label="Long term mean" )
#ax2.legend(loc="upper right"); plt.show()


# create a set of particles
n = 20  # number of particles
particles_list = []    # list of particles

#predict hidden variable V movement
for i in range(n):
    particle = Heston_process(mu=mu, rho=rho, sigma=sigma, theta=theta, kappa=kappa)
    V1 = particle.path(S0, v0, 3, 2/365)[1] #doesn't work when I use 2 steps instead of 3
    particles_list.append(V1[1]) 
#above appends list of n stocks' prices after 1 time step starting at S0

print(particles_list)

#measure step: reveal the true stock price at this time
z = Strue[1]
print('Strue measurement is ',z)

#find likelihood of Strue measurent given each of the particle's Vol

def measurementLikelihood(Y_t, Y , v, mu, dt):
    #gives the likelihood of the measurement given parameters ( p(Y_t|v_t, ... ) )
    return np.exp((-0.5*(Y+(mu-0.5*v)-Y_t)/(v*dt))**2)/ np.sqrt(2.0*np.pi*v*dt)

weights = []
for i in range(n):
    p = measurementLikelihood(z, S0 ,particles_list[i], mu, dt)
    weights[i] = p

print('weights =', weights)

#maybe make new class for particles





