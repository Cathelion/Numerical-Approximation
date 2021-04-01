#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 17:46:22 2021

This program interpolates Runge's function which
is notoriously bad behaved for equidistant grids
The zeros of chebychev polynomials as gridpoints are better

My own grid idea, where I decrease the number of points in the
middle, works better than the equigrid but not as good as
the chebychev grid.


@author: florian
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
plt.style.use('seaborn')

def f(x):
    return 1/(1+25*x**2)


def equi_grid(n):
    return np.linspace(-1,1,n)
    
def my_grid(n):
    # for n = 2m, does 2m-1
    p=int((n-1)/2)
    x = 2/((p+1)*p)
    data = [-1 + (i+1)*i*x/2 for i in range(p)]
    data2 = [-x for x in data]
    return np.array(data + [0] + data2)
 
def cheby_grid(n):    
    data = [np.cos((2*k+1)/ (2*n) * np.pi) for k in range(n)]
    return np.array(data)


def lagrange (x ,i , xm ):
    """
    Evaluates the i-th Lagrange polynomial at x
    based on grid data xm
    """
    n=len( xm )-1
    y=1 
    for j in range ( n+1 ):
        if i!=j:
            y*=( x-xm[j])/( xm[i]-xm[j])
    return y


def interpolation (x , xm , ym ):
    n=len( xm )-1
    lagrpoly = np.array ([ lagrange (x ,i , xm ) for i in range ( n+1 )])
    return np.dot( ym , lagrpoly )

def plot_int(gridtype):
    xm = np.linspace(-1,1,1000)
    ym = f(xm)
    plt.figure(figsize=(8,5))
    plt.plot(xm,ym)
    for n in [3,5,13]:
        xdat = gridtype(n)
        ydat = f(xdat)
        xm = np.linspace(-1,1,1000)
        ym = np.array([interpolation(x, xdat, ydat) for x in xm])    
        plt.plot(xm,ym,label='n = '+str(n))
    plt.legend()
    plt.ylabel('f(x)')
    plt.xlabel('x')

plot_int(equi_grid)
plot_int(my_grid)
plot_int(cheby_grid)


def plot_err(gridtype):
    xF = np.linspace(-1,1,1000)
    yF = f(xF)
    plt.figure(figsize=(8,5))
    for n in [3,5,13]:
        xdat = gridtype(n)
        ydat = f(xdat)
        xm = np.linspace(-1,1,1000)
        ym = np.array([interpolation(x, xdat, ydat) for x in xm])    
        plt.plot(xm,np.abs(ym-yF),label='n = '+str(n))
    plt.legend()
    plt.ylabel('error')
    plt.xlabel('x')
    
plot_err(equi_grid)
plot_err(my_grid)
plot_err(cheby_grid)

def compare_err():
    xF = np.linspace(-1,1,1000)
    yF = f(xF)
    plt.figure(figsize=(8,5))
    xdat = equi_grid(9)
    ydat = f(xdat)
    xm = np.linspace(-1,1,1000)
    ym = np.array([interpolation(x, xdat, ydat) for x in xm])    
    plt.plot(xm,np.abs(ym-yF),label='equidistant')
    xdat = my_grid(9)
    ydat = f(xdat)
    xm = np.linspace(-1,1,1000)
    ym = np.array([interpolation(x, xdat, ydat) for x in xm])    
    plt.plot(xm,np.abs(ym-yF),label='own grid')
    xdat = cheby_grid(9)
    ydat = f(xdat)
    xm = np.linspace(-1,1,1000)
    ym = np.array([interpolation(x, xdat, ydat) for x in xm])    
    plt.plot(xm,np.abs(ym-yF), label='Chebychev grid')
    plt.legend()
    plt.ylabel('error')
    plt.xlabel('x')
    
compare_err()
