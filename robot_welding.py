# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 03:06:41 2022

@author: Bassel
"""
import random

import numpy as np
import matplotlib.pyplot as plt
from spatialmath import *
import roboticstoolbox as rtb


#robot = rtb.models.Panda()
robot = rtb.models.Panda()

def create_points():
    r = 0.15
    #cx, cy, cz = 0, 0, 50
    #cx, cy, cz = np.asarray(robot.fkine(robot.qr-[0,0.5,0,0.5,0,0,0]))[:-1,-1]
    cx, cy, cz = np.asarray(robot.fkine(robot.qr))[:-1,-1]
    n_dots = 20
    rs = [r * random.randint(90,110)/100 for _ in range(n_dots)]
    hs = [random.randint(0,10)/100 for _ in range(n_dots)]
    points_in = np.array([
        [
            cx + r*np.sin(np.deg2rad(angle)),
            cy + r*np.cos(np.deg2rad(angle)),
            #50 + h
            cz
        ] for angle, r, h in zip(range(0, 360, int(360/n_dots)), rs, hs)
    ]).T
    return points_in


def sol(points):
    points_in=np.asarray(points)
    n_dots=points_in.shape[1]
    cond=points_in[0,0]-points_in[0,1]>0
    p1=np.array([points_in[0,0],points_in[1,0],points_in[2,0]])
    points_sh=np.insert(points_in.T,0,p1,axis=0).T
    points_extended=np.insert(points_in.T,n_dots,p1,axis=0).T
    points_vec=points_extended[:,1:]-points_sh[:,1:]
    mul=points_vec*points_vec
    #points_vec[2]=np.sqrt(mul[0]+mul[1])*np.tan(np.deg2rad(-45))
    points_vec[2]=np.sqrt(mul[0]+mul[1])*np.tan(np.deg2rad(-45))
    points_abs=np.linalg.norm(points_vec,axis=0)
    points_vec_normalized=points_vec/points_abs
    if cond:
        points_vec_normalized_rotated=np.array(-[points_vec_normalized[1],points_vec_normalized[0],points_vec_normalized[2]])
    else:
        points_vec_normalized_rotated=np.array([points_vec_normalized[1],-points_vec_normalized[0],points_vec_normalized[2]])
    out=np.insert(points_in, 3, points_vec_normalized_rotated,axis=0)
    out_aligned=np.array([out[0,:],out[1,:],out[2,:],out[5,:],out[4,:],out[3,:]])
    thx=np.ones(points.shape[1])*(0)
    thy=np.ones(points.shape[1])*(0)
    #thz=90-np.rad2deg(np.arctan2(points_vec_normalized_rotated[1],points_vec_normalized_rotated[0]))
    thz=np.ones(points.shape[1])*(0)
    return [out, [thx, thy, thz]]

table_coords = np.array([
    [-0.1,0.1,0.5],
    [0.1,0.1,0.5],
    [0.1,-0.1,0.5],
    [-0.1,-0.1,0.5]
])
table_plot = np.stack([
    table_coords[0],
    table_coords[1],
    table_coords[2],
    table_coords[3],
    table_coords[0],
    table_coords[2],
    table_coords[1],
    table_coords[3],
])



points_in=np.array([[ 4.84046815e-01,  5.30862890e-01,  5.69569570e-01,
         6.16321094e-01,  6.39544556e-01,  6.47546815e-01,
         6.28131878e-01,  6.00545263e-01,  5.69569570e-01,
         5.26227635e-01,  4.84046815e-01,  4.36767215e-01,
         3.99405739e-01,  3.56626639e-01,  3.39961753e-01,
         3.35546815e-01,  3.29975660e-01,  3.66334843e-01,
         4.02932451e-01,  4.33522537e-01],
       [ 1.54500000e-01,  1.44085062e-01,  1.17711973e-01,
         9.61028887e-02,  5.05242786e-02, -5.46483047e-17,
        -4.68160746e-02, -8.46410763e-02, -1.17711973e-01,
        -1.29819214e-01, -1.63500000e-01, -1.45511647e-01,
        -1.16498447e-01, -9.25761772e-02, -4.68160746e-02,
        -9.19387998e-17,  5.00607531e-02,  8.55227542e-02,
         1.11644345e-01,  1.55497740e-01],
       [ 4.12629775e-01,  4.12629775e-01,  4.12629775e-01,
         4.12629775e-01,  4.12629775e-01,  4.12629775e-01,
         4.12629775e-01,  4.12629775e-01,  4.12629775e-01,
         4.12629775e-01,  4.12629775e-01,  4.12629775e-01,
         4.12629775e-01,  4.12629775e-01,  4.12629775e-01,
         4.12629775e-01,  4.12629775e-01,  4.12629775e-01,
         4.12629775e-01,  4.12629775e-01]])

# points_in=np.array([[ 0.48404682,  0.74320972,  0.63099313,  0.33856997,  0.2700591 ],
#        [ 0.265     ,  0.08420713, -0.20225425, -0.20023171,  0.06952882],
#        [ 0.41262978,  0.41262978,  0.41262978,  0.41262978,  0.41262978]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# #ax.scatter(table_coords[:,0], table_coords[:,1], table_coords[:,2], c='g')
# #ax.plot(table_plot[:,0], table_plot[:,1], table_plot[:,2], c='g')

# ax.scatter(cx, cy, cz, c='r')
# ax.scatter(cx, cy, cz+20, c='r', alpha=0)
# ax.scatter(points_in[0], points_in[1], points_in[2], c='b')

#calculate end effector position and orientation
out, thl=sol(points_in)
v=50
# ax.scatter(out[0,0], out[1,0], out[2,0], c='k')
# ax.scatter(points_in[0,1], points_in[1,1], points_in[2,1], c='r')
q=np.zeros(7)
for i, dot in enumerate(out.T):
    cx, cy, cz, vx, vy, vz = dot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot([cx, cx+vx], [cy, cy+vy], [cz, cz+vz], c='r')
    T=SE3(cx, cy, cz) * SE3.RPY([vx, vy, vz], order='zyx')
    if robot.ik_lm_chan(T)[1]==1:
       ans=robot.ik_lm_chan(T)[0]
       if np.linalg.norm(robot.fkine(ans))<2.5:
          q=np.vstack((q, ans)) 
q=q[1:]                      
#robot.plot(q)
# cx, cy, cz, vx, vy, vz=out.T[20]
# T=SE3(cx, cy, cz) * SE3.RPY([vx, vy, vz], order='xyz')
#robot.plot(robot.ik_lm_chan(T)[0])
#robot.plot(robot.qr)

import swift
#import roboticstoolbox as rp
# import spatialmath as sm
# import numpy as np

env = swift.Swift()
env.launch(realtime=True)


robot.q = robot.qr
env.add(robot)
#robot.plot(q[0])

dt = 0.05
# v, arrived = rtb.p_servo(robot.fkine(robot.q), q[0], 1)
# robot.qd = np.linalg.pinv(robot.jacobe(robot.q)) @ v
# env.step(dt)


# #robot.q=robot.ik_lm_chan(T)[0]
# robot.q=q[-2]

# while not arrived:
#     robot.q=q[count]
#     count=count+1
#     env.step(dt)
#     if np.all(robot.q-q[-1]<0.01):
#         arrived=True

# cx, cy, cz, vx, vy, vz=out.T[1]
# T=SE3(cx, cy, cz) * SE3.RPY([vx, vy, vz], order='xyz')
# ans=robot.ik_lm_chan(T)[0]

# env.step(dt)
# robot.q=ans

# cx, cy, cz, vx, vy, vz=out.T[1]
# tq=robot.fkine(robot.qr)
# pq=np.asarray(tq)[:-1,3]
# T=SE3(pq[0], pq[1], pq[2]) * SE3.Ry(1) * SE3.Rz(1.4)
# #tq=tq*SE3.Ry(1)
# ans=robot.ik_nr(tq)[0]
# #ans=robot.qr

# env.step(dt)
# robot.q=ans
# env.step(dt)

k=-1
count_r=80
for i in range(points_in.T.shape[0]):
    print(i)
    cx, cy, cz, vx, vy, vz=out.T[i]
    #T=SE3(cx, cy, cz) * SE3.RPY([vx, vy, vz], order='xyz')
    T=SE3(cx,cy,cz)*SE3.RPY([thl[0][i], thl[1][i], thl[2][i]], unit='rad', order='xyz')
    arrived = False
    qd=np.zeros(7)
    count=0
    while not arrived:
        if count==count_r:
            break
        count=count+1
        v, arrived = rtb.p_servo(robot.fkine(robot.q), T, 2)
        robot.qd = np.linalg.pinv(robot.jacobe(robot.q)) @ v
        #qd=qd+robot.qd 
        env.step(dt)
    if k==-1:
        k=1
        count_r=2
    #robot.q= robot.q+robot.qd
