from math import *
import random
import matplotlib.pyplot as plt

# landmarks which can be sensed by the robot (in meters)
landmarks = [[20.0, 20.0], [20.0, 80.0], [20.0, 50.0],
             [50.0, 20.0], [50.0, 80.0], [80.0, 80.0],
             [80.0, 20.0], [80.0, 50.0]]
 
# size of one dimension (in meters)
world_size = 100.0

class robot:

    def __init__(self):
        #random.random() generates random number between 0 and 1
        self.x = random.random() * world_size           # robot's x coordinate
        self.y = random.random() * world_size           # robot's y coordinate
        self.orientation = random.random() * 2.0 * pi   # robot's orientation
 
        self.forward_noise = 0.0   # noise of the forward movement
        self.turn_noise = 0.0      # noise of the turn
        self.sense_noise = 0.0     # noise of the sensing

    def set(self, new_x, new_y, new_orientation):
        #Set robot's initial position and orientation 
        if new_x < 0 or new_x >= world_size:
            raise ValueError('X coordinate out of bound')
        if new_y < 0 or new_y >= world_size:
            raise ValueError('Y coordinate out of bound')
        if new_orientation < 0 or new_orientation >= 2 * pi:
            raise ValueError('Orientation must be in [0,...,2pi]')
        
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)

    def set_noise(self, new_forward_noise, new_turn_noise, new_sense_noise):
        #Set the noise parameters, changing them is often useful in particle filters 
        self.forward_noise = float(new_forward_noise)
        self.turn_noise = float(new_turn_noise)
        self.sense_noise = float(new_sense_noise)

    def sense(self):
        #Sense the environment: calculate distances to landmarks
        #Returns distances to landmarks
        z = []
        for i in range(len(landmarks)):
            dist = sqrt((self.x - landmarks[i][0])**2 + (self.y - landmarks[i][1])**2)
            dist += random.gauss(0.0, self.sense_noise)
            z.append(dist)
        return z

    def move(self, turn, forward):
        #Perform robot's turn and move
        #Returns robot's state after the move

        if forward < 0:
            raise ValueError('Robot cannot move backwards')

        # turn, and add randomness to the turning command
        orientation = self.orientation + float(turn) + random.gauss(0.0, self.turn_noise)
        orientation %= 2 * pi
        # move, and add randomness to the motion command
        dist = float(forward) + random.gauss(0.0, self.forward_noise)
        x = self.x + (cos(orientation) * dist)
        y = self.y + (sin(orientation) * dist)
        # cyclic boundaries (robot appears at opposite boundary if moves over edge)
        x %= world_size
        y %= world_size
        # set particle
        res = robot()
        res.set(x, y, orientation)
        res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
    
        return res
        #note, this doesn't change the input "self"

    def gaussian(self, mu, sigma, x):
        # mu is distance to landmark, x is distance to the landmark measured by the robot
        # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
        return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))
    
    def measurement_prob(self, measurement):
        #Calculate the measurement probability: how likely a measurement should be
        #measurement variable = current measurement
    
        prob = 1.0
        for i in range(len(landmarks)):
            dist = sqrt((self.x - landmarks[i][0]) ** 2 + (self.y - landmarks[i][1]) ** 2)
            prob *= self.gaussian(dist, self.sense_noise, measurement[i])
        return prob



walee = robot()
walee.set(30,10, 0)
print(walee.x, walee.y, walee.orientation)
walee = walee.move(pi/2, 10)
print(walee.x, walee.y, walee.orientation)
walee.set_noise(1, 1, 1)
walee = walee.move(0, 10)
print(walee.x, walee.y, walee.orientation)



#create robot with random pos and orientation
myrobot = robot()
z = myrobot.sense()
print('z = ', z)
print('myrobot = ', myrobot)

# create a set of particles
n = 1000  # number of particles
particles_list = []    # list of particles
 
for i in range(n):
    particle = robot()
    particle.set_noise(0.05, 0.05, 5.0)
    particles_list.append(particle)

#move robot and sense
myrobot = myrobot.move(0.1, 5.)
z = myrobot.sense()

# now we simulate a robot motion for each of
# these particles
particles_list2 = [] 
for i in range(n):
        particles_list2.append( particles_list[i].move(0.1, 5.) )

particles_list = particles_list2


w = []
for i in range(n):
    w.append(particles_list[i].measurement_prob(z))


# resampling with a sample probability proportional
# to the importance weight
p3 = []
 
index = int(random.random() * n)
beta = 0.0
mw = max(w)
 
for i in range(n):
    beta += random.random() * 2.0 * mw
 
    while beta > w[index]:
        beta -= w[index]
        index = (index + 1) % n
 
    p3.append(particles_list[index])
# here we get a set of co-located particles
particles_list = p3

def evaluation(r, p):
    sum = 0.0
    for i in range(len(p)):
        # the second part is because of world's cyclicity
        dx = (p[i].x - r.x + (world_size/2.0)) % \
             world_size - (world_size/2.0)
        dy = (p[i].y - r.y + (world_size/2.0)) % \
             world_size - (world_size/2.0)
        err = sqrt(dx**2 + dy**2)
        sum += err

    return sum / float(len(p))

print(', Evaluation = ', evaluation(myrobot, particles_list))



def visualization(robot, step, p, pr, weights):
 
    plt.figure('Robot in the world', figsize=(15., 15.))
    plt.title('Particle filter, step ' + str(step))
 
    # draw coordinate grid for plotting
    grid = [0, world_size, 0, world_size]
    plt.axis(grid)
    plt.grid(b=True, which='major', color='0.75', linestyle='--')
    plt.xticks([i for i in range(0, int(world_size), 5)])
    plt.yticks([i for i in range(0, int(world_size), 5)])
 
    # draw particles
    for ind in range(len(p)):
 
        # particle
        circle = plt.Circle((p[ind].x, p[ind].y), 1., facecolor='#ffb266', edgecolor='#994c00', alpha=0.5)
        plt.gca().add_patch(circle)
 
        # particle's orientation
        arrow = plt.Arrow(p[ind].x, p[ind].y, 2*cos(p[ind].orientation), 2*sin(p[ind].orientation), alpha=1., facecolor='#994c00', edgecolor='#994c00')
        plt.gca().add_patch(arrow)
 
    # draw resampled particles
    for ind in range(len(pr)):
 
        # particle
        circle = plt.Circle((pr[ind].x, pr[ind].y), 1., facecolor='#66ff66', edgecolor='#009900', alpha=0.5)
        plt.gca().add_patch(circle)
 
        # particle's orientation
        arrow = plt.Arrow(pr[ind].x, pr[ind].y, 2*cos(pr[ind].orientation), 2*sin(pr[ind].orientation), alpha=1., facecolor='#006600', edgecolor='#006600')
        plt.gca().add_patch(arrow)
 
    # fixed landmarks of known locations
    for lm in landmarks:
        circle = plt.Circle((lm[0], lm[1]), 1., facecolor='#cc0000', edgecolor='#330000')
        plt.gca().add_patch(circle)
 
    # robot's location
    circle = plt.Circle((robot.x, robot.y), 1., facecolor='#6666ff', edgecolor='#0000cc')
    plt.gca().add_patch(circle)
 
    # robot's orientation
    arrow = plt.Arrow(robot.x, robot.y, 2*cos(robot.orientation), 2*sin(robot.orientation), alpha=0.5, facecolor='#000000', edgecolor='#000000')
    plt.gca().add_patch(arrow)
 
    plt.savefig('figure_' + str(step) + '.png')
    plt.close()

