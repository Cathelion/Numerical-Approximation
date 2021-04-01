#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 17:02:12 2021

@author: florian

This program deals with Hermite interpolation,
i.e. interpolation such that higher derivatives 
match up too. 
Using divided differences



"""

import numpy as np

datX = np.array([0,0,np.pi,np.pi,np.pi,2*np.pi])
datY = np.array([1,0,0,-1/np.pi, 2/np.pi**2,0])

n = len(datX)
datM =  np.zeros((n+1,n))
datM[0] = datX

# fill with function values
for i in range(n):
    datM[1][i] = datY[np.argwhere(datX==datM[0,i])[0,0]]

# divided differences algorithm
for k in range(2,n+1):  # over columns 
    
    for j in range(k-1,n):  #over rows
        
        start = datM[0,j-k+1]
        end = datM[0,j]
        
        #take the provided derivative
        if start==end: 
            datM[k,j] = datY[np.argwhere(datX==start)[k-1,0]]/np.math.factorial(k-1)
        
        # compute divided differences
        else: 
            datM[k,j] = (datM[k-1,j]-datM[k-1,j-1])/(end-start)
    
print(datM)
coeff = np.diagonal(datM,-1)
print("coefficiants:" ,coeff)


def newtonPoly(x,nodes,k):
    nPoly = 1
    for i in range(k):
        nPoly *= (x-nodes[i])
    return nPoly

def hermite(x,coeff):
    res=0
    for i in range(len(coeff)):
        res += coeff[i]

xPl=np.linspace(0, 2*np.pi,100)





