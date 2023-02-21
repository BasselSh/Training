# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 06:06:11 2023

@author: Bassel
"""

import numpy as np
import matplotlib.pyplot as plt

def showSeg(p,i):
    plt.figure()
    plt.scatter(p.T[0,:],p.T[1,:])
    plt.plot(p[:i+1,0], p[:i+1,1])
    if i+2<p.shape[0]:
        plt.plot(p[i:i+2,0], p[i:i+2,1])


def convex(p,i):
    global flag
    global total
    global count
    if i+1==p.shape[0] and flag==-1: # ending the convex
        plt.scatter(p0.T[0,:],p0.T[1,:])
        plt.plot(p[:,0], p[:,1])
        return 1
    if i+1==p.shape[0] and flag: #ending of the first move to the right
        flag=-1
        total.sort(key=lambda x:x[0],reverse=True)
        total.append(start)
        p=np.vstack((p,total))
        
    l1=p[i]-p[i-1]
    l2=p[i+1]-p[i-1]
    showSeg(p,i)
    sign=np.cross(l1,l2)
    
    if sign >=0:
        convex(p,i+1)
    else:
        if p[i-1,0]==start[0] and p[i-1,1]==start[1]:
            total.append(p[i])
            p=np.delete(p,i,axis=0)
            return convex(p,i)
        else:
            total.append(p[i])
            p=np.delete(p,i,axis=0)
            return convex(p,i-1)


p=np.random.rand(20,2)

p=np.asarray(sorted(p,key= lambda x:x[0,]))
plt.scatter(p.T[0,:],p.T[1,:])

p0=p
start=p0[0]
total=[]
flag=1

plt.show()
print(convex(p0,1))
print("everything is fine")