import numpy as np
import math
import scipy.stats as ss
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt


class SV_process():
    #x[t] is state: stochastic volatility path
    #y[t] is measurement: log of price returns ( log(S[t]/S[t-1]) )
    #note: this model is without stock price drift

    def __init__(self, mu, sigma, kappa):
        #mu =  long term mean
        #sigma = vol of process (ie. vol of vol)
        #kappa = rate of revergence (below 1 otherwise diverges)
        self.mu = mu
        self.sigma = sigma
        self.kappa = kappa
    
    def path(self, xnoisemean, xnoisevar, ynoisemean, ynoisevar, N):
        #Produces x SV process path
        #y is measurement path
        #N = number of time steps
        #T = total time
        dt = T/N
        epsX = np.random.normal(xnoisemean, xnoisevar, N)
        epsY = np.random.normal(ynoisemean, ynoisevar, N)
       
        x0 = np.random.normal(self.mu, xnoisevar/(1 - (self.sigma)**2))
        x = np.zeros(N)
        x[0] = x0
        y = np.zeros(N)
        y[0] = math.exp(x[0]/2) * epsY[0]

        # Generate paths
        for t in range(1, N):
            x[t]= self.mu + self.kappa*(x[t-1]-self.mu) + self.sigma*epsX[t]
            y[t]= math.exp(x[t]/2) * epsY[t]
        return x, y

#model params

N = 50
T = 50
mu = 0
sigma = 0.7
kappa = 0.5
xnoisemean = 0
xnoisevar = 0.02
ynoisemean = 0
ynoisevar = 0.04

model = SV_process(mu, sigma, kappa)
modelpath = model.path(xnoisemean, xnoisevar, ynoisemean, ynoisevar, N)
truestate = modelpath[0]
measurement = modelpath[1]
Tvec = np.linspace(0, T, N, True)



#Particle Filter
n = 20 #starting number of particles
particles = np.zeros((n,N))
weights = np.zeros((n,N))

#initial time step

for i in range(0, n):
    particles[i,0] = np.random.normal(mu, xnoisevar/(1 - (sigma)**2))
    weights[i,0] = math.exp(-0.5*( (measurement[0]**2)*math.exp(-particles[i,0]) - particles[i,0]))
    if i==n-1:
        totalweight = np.sum(weights[:,0])
        for j in range(0,n):
            weights[j,0] = weights[j,0]/totalweight

#resampling code

#cumsum = np.cumsum(weights[:,0])
#newweights = []
#new = np.zeros(n)
#for i in range(0,n-1):
    #random = np.random.uniform(0, 1)
    #for j in range(0,n-1):
        #if random < cumsum[0]:
            #newweights.append(weights[i])
        #elif cumsum[j] < random < cumsum[j+1]:
            #newweights.append(weights[i+1])

#alternative way to do resampling
sampledparticles = np.random.choice(particles[:,0], n, True, weights[:,0])

#rest of the time steps
for t in range(1, N):
    for i in range(0, n):
        #update each particle with process movement
        particles[i,t]= mu + kappa*(particles[i,t-1]-mu) + sigma * np.random.normal(0,xnoisevar)
        #collect (noisy) measurement of true state NEED TO CHANGE THIS
        y = measurement[t]
        #assign weights to each particle
        weights[i,t] = math.exp(-0.5*( (y**2)*math.exp(-particles[i,t]) - particles[i,t]))
        if i==n-1:
            totalweight = np.sum(weights[:,t])
            for j in range(0,n):
                weights[j,t] = weights[j,t]/totalweight
                # making sure weights add up to 1
    sampledparticles = np.random.choice(particles[:,t], n, True, weights[:,t])
    particles[:,t] = sampledparticles

print(particles)
print(weights)


#plotting particles
particlesplotx = np.zeros(n*N)
particlesploty = np.zeros(n*N)

for t in range(0, N):
    for i in range(0, n):
        particlesplotx[t*n + i] = t
        particlesploty[t*n + i] = particles[i,t]

plt.scatter(particlesplotx, particlesploty)
plt.plot(Tvec, truestate, 'b')
plt.plot(Tvec, measurement, 'r')
plt.show()


#tracking accuracy of the PF
error = np.zeros((n,N))
errorT = np.zeros(N)
for t in range(0,N):
    for i in range(0,n):
        error[i,t] = abs(truestate[t]-particles[i,t])
    errorT[t] = np.sum(error[:,t])

plt.plot(Tvec, errorT, 'g')
plt.show()
