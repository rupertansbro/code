import numpy as np
import math
import scipy.stats as ss
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt


class AR1_process():
    #called AR(1) cause its only dependent only 1 timestep back

    def __init__(self, mu, theta):
        #mu =  long term mean
        #theta = mean reversion rate
        self.mu = mu
        self.theta = theta         
    
    def path(self, xnoisemean, xnoisevar, ynoisemean, ynoisevar, N):
        #Produces x AR1 process path
        #y is measurement path
        #N = number of time steps
        #T = total time
        dt = T/N
        epsX = np.random.normal(xnoisemean, xnoisevar, N)
        epsY = np.random.normal(ynoisemean, ynoisevar, N)
       
        x0 = np.random.normal(self.mu, xnoisevar/(1 - (self.theta)**2))
        x = np.zeros(N)
        x[0] = x0
        y = np.zeros(N)
        y[0] = x0 + epsY[0]

        # Generate paths
        for t in range(1, N):
            x[t]=self.mu*(1-self.theta) + self.theta*x[t-1] + epsX[t]
            y[t]=x[t] + epsY[t]
        return x, y

N = 40
T = 40
mu = 5
theta = 0.975
xnoisemean = 0
xnoisevar = 0.02
ynoisemean = 0
ynoisevar = 0.05

model = AR1_process(mu, theta)
modelpath = model.path(xnoisemean, xnoisevar, ynoisemean, ynoisevar, N)
truestate = modelpath[0]
measurement = modelpath[1]
Tvec = np.linspace(0, T, N, True)


#Kalman Filter
# KF requires dynamic model and measurement model to be linear
# and also for noise to be gaussian with 0 mean
# so works for this AR(1) model
Xest = np.zeros(N)
P = np.zeros(N)
K = np.zeros(N)
Xest[0] = mu #initial state estimate
P[0] = xnoisevar/(1 - theta**2) #initial estimate uncertainty

for t in range(0, N-1):
    #step 1: measure
    y = measurement[t]
    #step 2: calculate kalman gain
    K[t] = P[t]/(P[t] + ynoisevar)
    #step 3: update state estimate
    Xest[t] = Xest[t] + K[t]*(y - Xest[t])
    #step 4: update estimate uncertainty
    P[t] = P[t]*(1 - K[t])
    #step 5: predict next state
    Xest[t+1] = mu*(1-theta) + theta*Xest[t]
    P[t+1] = P[t] + xnoisevar
    #step 6: t+1 becomes t

plt.plot(Tvec, measurement, 'r')
plt.plot(Tvec, truestate, 'b') 
plt.plot(Tvec, Xest, 'g') 
plt.show()


#Particle Filter
n = 20 #starting number of particles
particles = np.zeros((n,N))
weights = np.zeros((n,N))

#initial time step

for i in range(0, n):
    particles[i,0] = np.random.normal(mu, xnoisevar/(1 - (theta)**2))
    weights[i,0] = math.exp(-0.5*( (measurement[0] - particles[i,0])**2 / xnoisevar))/np.sqrt(2*np.pi*xnoisevar)
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
        particles[i,t]= mu*(1-theta) + theta*particles[i,t-1] + np.random.normal(0,xnoisevar)
        #collect (noisy) measurement of true state
        y = measurement[t]
        #assign weights to each particle
        weights[i,t] = math.exp(-0.5*( (y - particles[i,t])**2 / xnoisevar))/np.sqrt(2*np.pi*xnoisevar)
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
