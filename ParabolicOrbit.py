# from sympy import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
rp = 1.496*10**11 # Radius periapsis in meters (1AU = 1.496e11m)
M = 1.989*10**30 # Mass of sun in kg
G = 6.67430*10**-11 # gravitational constant N*m^2*kg^-2

n_tenday = 36 # number of 10 day periods before and after periapsis
t_tenday= 864000 # 10 days in seconds

# For animation use:
n_frames = 100
t = np.linspace(-n_tenday*t_tenday,n_tenday*t_tenday,n_frames)

# For still plot use:
# t = np.arange(-n_tenday*t_tenday,n_tenday*t_tenday,t_tenday)

# Newtons method where
# psi = eccentric anomaly
# phi = true anomaly
# r = position
def f(psi):
    return t-np.sqrt(2*rp**3/(M*G))*(psi+1/3*psi**3)


def derivative(f, x, h):
    return (f(x + h / 2.0) - f(x - h / 2.0)) / h


def root(f, x0, h, cycles):
    for i in range(cycles):
        x1 = x0 - f(x0) / derivative(f, x0, h)
        x0 = x1
    return x0


psi = root(f, 1.0, 0.001, 400) # Find roots

r = rp*(1+psi**2) # Find the magnitude of the position vector r from psi
phi = np.arccos(2*rp/r-1) # Find true anomaly phi from r and rp

x = r*np.cos(phi)   # express x, y coordinates in terms of r and phi
y = r*np.sin(phi)

x_au = x/(1.496*10**11) # Convert x,y into AU
y_au = y/(1.496*10**11)


y_au[int(len(t)/2):] = -y_au[int(len(t)/2):] # Sign change the all y-values after periapsis

sun = plt.Circle((0, 0), 0.1, color='gold')

# Still Plot
# fig, ax = plt.subplots()
# ax.set_aspect('equal', adjustable='box') # Change aspect ratio to square
# ax.axis([-3,3,-3,3]) # adjust (current axis may not show the entire specified time period n_tenday)
# plt.title('Parabolic Path of a Comet around the Sun ({} days)'.format(str(n_tenday*2)))
# plt.xlabel('x (AU)')
# plt.ylabel('y (AU)')
# plt.plot(x_au, y_au,c ='k', linewidth=2)
# plt.plot(x_au, y_au, marker='o', c='k', markersize=4)  # Plot 10-day periods as points
# ax.add_patch(sun)
# label = ax.annotate("Sun", xy=(0, 0.2), fontsize=14, ha="center")
# plt.show()

# Animation
fig,ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
plt.title('Parabolic Path of a Comet around the Sun ({} days)'.format(str(n_tenday*2)))
plt.plot(x_au, y_au, 'k-', linewidth=1) # plot still trajectory
point, = ax.plot(x_au[0],y_au[0], marker="o",c='r') # Plot first point. From here the animation will update this point
plt.xlabel('x (AU)')
plt.ylabel('y (AU)')
# ax.axis([-0.7,1.5,-3,3])
ax.axis([-3,3,-3,3])
ax.add_patch(sun)
label = ax.annotate("Sun", xy=(0, 0.2), fontsize=14, ha="center")

def position(i): # return x,y position from the index in n_frames
    return np.array([x_au[i],y_au[i]])

def update(i): # update x,y position of point
    x,y = position(i)
    point.set_data([x], [y])
    return point,

# See https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
anim = FuncAnimation(fig, update, interval=20, blit=True, repeat=True,frames=np.arange(1,n_frames,1))

plt.show()

anim.save('parabolic_orbit.gif')
