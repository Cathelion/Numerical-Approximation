#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 16:49:54 2021

@author: florian

This program implements the clenshaw algorithm 
and computes the fourier coefficients with the fft

The clenshaw algorithm is used to compute the partial sums
of chebychev polynomials, which can be used to approximate 
a function in wrt an inner product norm.

"""
import numpy as np
import scipy.fft as fft
import scipy.integrate as sci
import matplotlib.pyplot as plt
from matplotlib import style
plt.style.use('seaborn')
plt.figure(figsize=(8,6))



def clenshaw(a,x):
    z = np.array([a[-1],0,0])
    for k in range(0,len(a)-1):
        z[2] = z[1]
        z[1] = z[0]
        z[0] = a[-(k+2)] + 2*x*z[1] - z[2]
    return (z[0]-z[2])/2


def manual_coeff(f,n):
    tmp2 = lambda x: 1
    a_0 = 2* inner_prod(f, tmp2) / inner_prod(tmp2, tmp2)
    coeff = [a_0]
    for k in range(1,n):
        a_k = one_coeff(f,k) 
        coeff.append(a_k)
    return coeff

def one_coeff(f,k):
    T_k = lambda x: np.cos(k*np.arccos(x))
    return 2/np.pi * inner_prod(f, T_k)

def inner_prod(f,g):
    def lcl1(x):
        return f(x)*g(x)*1/np.sqrt(1-x**2)
    return sci.quad(lcl1,-1,1)[0]

def plot_fft(func,n):
    xRef = np.linspace(-1,1,1000)
    coeff = a_n_fft(func,n)
    yDat5= np.array([clenshaw(coeff,x) for x in xRef])
    #plt.plot(xRef, yDat5)
    return yDat5


def plot_quad(func,n):
    xRef = np.linspace(-1,1,1000)
    coeff = manual_coeff(func, n)
    yDat4= np.array([clenshaw(coeff,x) for x in xRef])
    #plt.plot(xRef, yDat4)
    return yDat4

def a_n_fft(func,n):
    xDat = np.linspace(0,2*np.pi,n)
    yDat =func(np.cos(xDat))
    coeff_fft = fft.fft(yDat)
    rev = np.hstack((np.array(coeff_fft[0]), np.flip(coeff_fft[n//2+1:])))
    return np.real (coeff_fft[0:n//2]  + rev)/n

def err_fft(func,n):
    xRef = np.linspace(-1,1,1000)
    yDat_fft = plot_fft(func,n)
    yRef = np.array([func(x) for x in xRef])
    return np.abs(yDat_fft-yRef)

def err_quad(func,n):
    xRef = np.linspace(-1,1,1000)
    yDat = plot_quad(func,n)
    yRef = np.array([func(x) for x in xRef])
    return np.abs(yDat-yRef)

def fftplotting():
    xRef = np.linspace(-1,1,1000)
    func = lambda x:np.arctan(x)
    for n in [16,32,64,128]:
        plt.figure(1)
        plt.plot(xRef,plot_fft(func, n), label="approximation with n="+str(n))
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.figure(2)
        plt.plot(xRef,err_fft(func,n),label="error with n="+str(n))
        plt.xlabel("x")
        plt.ylabel("absolute error")
        plt.legend()
        
    plt.figure(1)
    plt.plot(xRef,func(xRef),label="original function")
    plt.legend()

def quadplotting():
    xRef = np.linspace(-1,1,1000)
    func = lambda x:np.arctan(x)
    for n in [3,5,11,25]:
        plt.figure(1)
        plt.plot(xRef,plot_quad(func, n), label="approximation with n="+str(n))
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.figure(2)
        plt.plot(xRef,err_quad(func,n),label="error with n="+str(n))
        plt.xlabel("x")
        plt.ylabel("absolute error")
        plt.legend()
        
    plt.figure(1)
    plt.plot(xRef,func(xRef),label="original function")
    plt.legend()

xRef = np.linspace(-1,1,1000)
n=100
func = lambda x:np.abs(x)
plt.figure(1)
plt.plot(xRef,func(xRef),label="original function")
plt.plot(xRef,plot_quad(func, n), label="approximation with n="+str(n))
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()

plt.figure(2)
plt.plot(xRef,func(xRef),label="original function")
plt.plot(xRef,plot_fft(func, n), label="approximation with n="+str(n))
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
