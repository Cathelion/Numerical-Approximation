#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 19:30:30 2021

@author: florian

This porgram approximates the arctan function by
using legendre polynomials, an orthogonal system 
of polynomials.


"""
import numpy as np
import scipy.integrate as sc
import matplotlib.pyplot as plt
from matplotlib import style
plt.style.use('seaborn')
plt.figure(figsize=(10,7))

def binom(n,k):
    return np.math.factorial(n)//(np.math.factorial(k)*np.math.factorial(n-k))

# nth legendre polynomial at x
def legend(n,x):
    res =0
    for k in range(n+1):
        res += binom(n,k)* binom(n+k, k) * ((x-1)/2)**k
    return res

def RHS(n):
    rhs = np.zeros(n)
    for i in range(n):
        def lcl_fcn(x):
            return legend(i,x)*np.arctan(x)
        rhs[i]=sc.quad(lcl_fcn,-1,1)[0]
    return rhs

def norm_legend(n):
    return np.array([2/(2*i+1) for i in range(n)])

def coeff(n):
    return np.divide(RHS(n),norm_legend(n))

def best_approx(x,n):
    coef = coeff(n)
    res=0
    for i in range(n):
        res += legend(i,x)*coef[i]
    return res

def error(x,n):
    return np.abs(best_approx(x,n)-np.arctan(x))

def plot_approx():
    xDat = np.linspace(-1,1,200)
    plt.plot(xDat,np.arctan(xDat),label="arctan(x)")
    for i in [2,4,6]:
        def lcl_best_approx(x):
            return best_approx(x,i)
        lcl_best_approx=np.vectorize(lcl_best_approx)
        plt.plot(xDat,lcl_best_approx(xDat),label="n = "+str(i))
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")

def plot_error():
    xDat = np.linspace(-1,1,200)
    for i in [2,4,6]:
        def lcl_error(x):
            return error(x,i)
        lcl_error=np.vectorize(lcl_error)
        plt.plot(xDat,lcl_error(xDat),label="n = "+str(i))
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")

plot_approx()
#plot_error()
