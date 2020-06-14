"""Light Test

An elaborate benchmark test for SunLib.py
"""
import copy
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import SunLib
import numpy as np

# create  locus of sun path
arcRad = 45
x_start, x_end = -arcRad, arcRad
num_suns = 100
suns_x = np.linspace(x_start, x_end, num_suns)
suns_y = np.sqrt(arcRad**2 - suns_x**2)

# initialize. this variable will hold all edges
myEdges = []

# base building
bld1 = SunLib.BuildingsToEdges([[[-10,0],[10,0],[10,10],[-10,10]]])
myEdges.extend(bld1)

# 2nd building
bld2 = SunLib.BuildingsToEdges([[[-5,10],[5,10],[5,15],[-5,15]]])
myEdges.extend(bld2)

# jewel
jewel = SunLib.BuildingsToEdges([[[-2,17],[2,17],[2.5,19],[0,20], [-2.5, 19]]])
myEdges.extend(jewel)

# jewel guards
myEdges.append([[-3.5,17], [-3.5, 25]])
myEdges.append([[3.5,17], [3.5, 25]])

# tower 1
twr1 = SunLib.BuildingsToEdges([[[-18,0],[-15,0],[-15,15],[-18,15]]])
myEdges.extend(twr1)

# tower 2
twr2 = SunLib.BuildingsToEdges([[[18,0],[15,0],[15,15],[18,15]]])
myEdges.extend(twr2)

# add top triangle
tri1 = SunLib.BuildingsToEdges([[[-2,11],[2,11],[0,13]]])
myEdges.extend(tri1)

# add floor
myEdges.append([[-40,0], [40,0]])

#litEdges = SunLib.GiveLitEdges(Mybuildings, [suns_x[0], suns_y[0]])
litEdges = SunLib.GiveIlluminatedEdges(myEdges, [suns_x[0], suns_y[0]])

# size of axis limit and location of text on plots
domainFactor = 1.3
textFactor = 0.9

# create figure and grab fig, axis handles
fig, ax = plt.subplots(figsize=[12,6])
ax.clear()
ax.set_facecolor('k')

# plot sun
pt = ax.scatter(suns_x[0], suns_y[0], s = 200, color='w')

# plot all edges
for edge in myEdges:
    ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], color='r')

#plot lit edges
for edge in litEdges:
    ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], color='w')

def animate(n):
    ax.clear()
    #ax.scatter(mySuns[n][0], mySuns[n][1], s = 200, color = 'r')
    ax.set_xlim(-domainFactor*arcRad, domainFactor*arcRad)
    ax.set_ylim(-10, domainFactor*arcRad)

    ax.scatter(suns_x[n%num_suns], suns_y[n%num_suns], s = 200, color = 'w')

    # plot all edges
    for edge in myEdges:
        ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], color='r')

    litEdges = SunLib.GiveIlluminatedEdges(myEdges, [suns_x[n%num_suns], suns_y[n%num_suns]])

    litLen = round(SunLib.EdgesLen(litEdges), 2)
    ax.text(textFactor*arcRad, textFactor*arcRad, "illuminated length\n         {}".format(litLen), color='w')

    #plot lit edges
    for edge in litEdges:
        ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], color='w')

ani = animation.FuncAnimation(fig, animate, interval=100, save_count=num_suns)
ani.save("MoonTest.mp4")

plt.show()
