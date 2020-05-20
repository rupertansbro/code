import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform, randn

#for tracking pos and direction of robot
#particles have vectors (x-pos, y-pos, direction in radians)

def create_uniform_particles(x_range, y_range, hdg_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N) # x[n] takes the n'th element of x
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
    particles[:, 2] %= 2 * np.pi # %= is modulus
    return particles

def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    particles[:, 2] = mean[2] + (randn(N) * std[2])
    particles[:, 2] %= 2 * np.pi
    return particles

gaussianPoints = create_gaussian_particles((60,170,np.pi), (3,10,0.1), 100)
plt.scatter(gaussianPoints[:,0],gaussianPoints[:,1]) 
plt.show()
#just 2d for now, dont know how to show direction yet

# update step: enter a control input step u to give
#the robot a command
def predict(prevParticles, u, std, dt=1.):
    # control input: u(direction change, speed)
    # noise Q (std direction change, std speed)`"""
    particles = np.copy(prevParticles)
    N = len(particles)
    # update direction of each particles
    particles[:, 2] += u[0] + (randn(N) * std[0])
    particles[:, 2] %= 2 * np.pi
    # move in the (noisy) commanded direction
    dist = (u[1] * dt) + (randn(N) * std[1])
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist
    return particles

predictedPoints = predict(gaussianPoints, [np.pi/2, 4], [0.1, 0.4], 1)

print(gaussianPoints)
print(predictedPoints)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(gaussianPoints[:,0],gaussianPoints[:,1], c='b', label='initial particles')
ax1.scatter(predictedPoints[:,0],predictedPoints[:,1], c='r', label='predicted movement')
plt.legend(loc='upper left')
plt.show()
#just 2d for now, dont know how to show direction yet


"""def update(particles, weights, z, R, landmarks):
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        #a[0:2] means from index 0 to 2, not including 2
        weights *= scipy.stats.norm(distance, R).pdf(z[i])

    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize"""

"""def estimate(particles, weights):
    #returns mean and variance of the weighted particles

    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var"""