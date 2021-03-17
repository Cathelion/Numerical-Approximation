#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 20:16:53 2021

@author: florian

A short demonstration of the Remez algorithm to obtain the best
approximation of a function in the supremum norm on a compact interval.

The test function used is f(x) = x*sin(x) and we use 7 points

We also plot the sign-changing error a characteristic of the approximation in the supremum norm,
and its convergence. Furthermore we keep track of the set of extremal points.

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
plt.style.use('seaborn')
plt.figure(1,figsize=(10,7))
plt.figure(2,figsize=(10,7))
plt.figure(1)


def coeff(xDat,func):
    n = len(xDat)-2
    powerM = np.tile( np.arange(n+1) ,(n+2,1))
    dataM = np.tile(np.reshape(xDat, (-1, 1)) , n+1)
    Mat= np.power(dataM,powerM)
    eta_vec = np.array([-(-1)**i for i in range(n+2)])
    Mat = np.hstack((Mat,np.reshape(eta_vec, (-1, 1))))
    b  = func(xDat)
    a = np.linalg.solve(Mat, b)
    return a

def evalPoly(coeff,x):
    n = len(coeff)
    powerV = np.array([x**i for i in range(n)])
    return np.dot(powerV,coeff)

def findxhat(coeff,func,a,b):
    xD = np.linspace(a,b,1000)
    yD = np.array([evalPoly(coeff, x) for x in xD])
    error = np.abs(yD-func(xD))
    return xD[error.argmax()]

def exchange(xDat, xNew, coeff, func):
    sgn_hat = np.sign(evalPoly(coeff, xNew)-func(xNew))

    if xNew<xDat[0]:
        sgn_x0 = np.sign(evalPoly(coeff, xDat[0])-func(xDat[0]))
        if sgn_x0==sgn_hat:
            xDat[0] = xNew
        else:
            xDat[-1] = xNew
        return
    
    elif xNew>xDat[-1]:
        sgn_x0 = np.sign(evalPoly(coeff, xDat[-1])-func(xDat[-1]))
        if sgn_x0==sgn_hat:
            xDat[-1] = xNew
        else:
            xDat[0] = xNew
        return
    
    for k in range(len(xDat)-1): 
        if xDat[k]<=xNew<=xDat[k+1]:
            sgn_xk = np.sign(evalPoly(coeff, xDat[k])-func(xDat[k]))
            if sgn_xk==sgn_hat:
                xDat[k] = xNew
            else:
                xDat[k+1] = xNew
            return
    xDat.sort()   # sort data in case new value in start/end
    
def remez(xDat,f,a,b):
    errors = []
    maxErrorOld = 0
    E_k_sets = [["%.4f" % x for x in xDat]]
    for i in range(20):
        coeff1 = coeff(xDat,f)
        maxErrorNew = coeff1[-1]
        coeff1=coeff1[:-1]
        
        if i%2==0:
            plt.figure(1)
            xD = np.linspace(a,b,5000)
            yD = [evalPoly(coeff1, x) for x in xD]
            plt.plot(xD,yD,label="i = "+str(i),linewidth=0.8)
            plt.xlabel("x")
            plt.ylabel("f(x)")
            
            plt.figure(2)
            yD2 = [evalPoly(coeff1, x)-f(x) for x in xD]
            plt.plot(xD,yD2, label="i = "+str(i),linewidth =1)
            plt.xlabel("x")
            plt.ylabel("error")
            
        xHat = findxhat(coeff1, f,a,b) 
        exchange(xDat, xHat, coeff1, f)
        E_k_sets.append(["%.4f" % x for x in xDat])
        
        errors.append(np.abs(maxErrorNew-maxErrorOld))
        
        if np.abs(maxErrorOld-maxErrorNew)<10**(-12):
            plt.figure(3)
            plt.scatter(list(range(len(errors))), errors)
            plt.xlabel("number of iterations")
            plt.ylabel("absolute change in error")
            print("DONE!   max error is ", np.abs(maxErrorNew))
            print("Polynomial coeffs are ", coeff1  )
            print("E_k: \n", E_k_sets)
            break
        else:
            maxErrorOld = maxErrorNew

a,b=0,2*np.pi
test = np.linspace(a,b,7)
f = lambda x: x*np.sin(x)  

xD = np.linspace(a,b,1000) 
plt.plot(xD,f(xD))

remez(test,f,a,b)

plt.figure(2)
plt.legend()

plt.figure(1)
plt.legend()