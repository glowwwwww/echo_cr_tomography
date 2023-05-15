import os
import time
import datetime
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy import optimize
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.stats import rv_discrete
from matplotlib import pyplot as plt
from numpy.random import default_rng
import numdifftools as nd

# z,y,x

U_label= [r"$(U_0, X)$",r"$(U_0, Y)$",r"$(U_0, Z)$",r"$(U_1, X)$",r"$(U_1, Y)$",r"$(U_1, Z)$"]

def rabi_x(t, lam0):
    mu  = lam0[0]*1e6
    a   = lam0[1]*1e6
    b   = lam0[2]*1e6
    phi = lam0[3]
    return -a*b/mu**2*np.cos(phi)*(1-np.cos(2*mu*t))-b/mu*np.sin(phi)*np.sin(2*mu*t)

def rabi_y(t, lam0):
    mu  = lam0[0]*1e6
    a   = lam0[1]*1e6
    b   = lam0[2]*1e6
    phi = lam0[3]
    return -a*b/mu**2*np.sin(phi)*(1-np.cos(2*mu*t))-b/mu*np.cos(phi)*np.sin(2*mu*t)

def rabi_z(t, lam0):
    mu  = lam0[0]*1e6
    b   = lam0[2]*1e6
    return (1-b**2/mu**2)+b**2/mu**2*np.cos(2*mu*t)


def rabi_xt(lam0):
    t   = lam0[0]
    mu  = lam0[1]*1e6
    a   = lam0[2]*1e6
    b   = lam0[3]*1e6
    phi = lam0[4]
    return -a*b/mu**2*np.cos(phi)*(1-np.cos(2*mu*t))-b/mu*np.sin(phi)*np.sin(2*mu*t)

def rabi_yt(lam0):
    t   = lam0[0]
    mu  = lam0[1]*1e6
    a   = lam0[2]*1e6
    b   = lam0[3]*1e6
    phi = lam0[4]
    return -a*b/mu**2*np.sin(phi)*(1-np.cos(2*mu*t))-b/mu*np.cos(phi)*np.sin(2*mu*t)

def rabi_zt(lam0):
    t   = lam0[0]
    mu  = lam0[1]*1e6
    a   = lam0[2]*1e6
    b   = lam0[3]*1e6
    phi = lam0[4]
    return (a**2/mu**2)+b**2/mu**2*np.cos(2*mu*t)+phi*0

def grad_lam(t, mu, a, b, phi, n_data):
    n_rabi = n_data%3
    if n_rabi ==0:
        return  nd.Jacobian(rabi_xt)([t, mu, a, b, phi])
    elif n_rabi ==1:
        return  nd.Jacobian(rabi_yt)([t, mu, a, b, phi])
    elif n_rabi == 2:
        return  nd.Jacobian(rabi_zt)([t, mu, a, b, phi])

def gradient_rabi(t, mu0, a0, b0, phi0, mu1, a1, b1, phi1, n_data):
    n_rabi = n_data%3
    if n_data//3==0:
        dlam = np.array([grad_lam(t, mu0, a0, b0, phi0, n_rabi)])
        dlam = np.array(dlam[0][0])
        return np.array([*dlam, 0, 0, 0, 0])
    elif n_data//3==1:
        dlam = np.array([grad_lam(t, mu1, a1, b1, phi1, n_rabi)])
        dlam = np.array(dlam[0][0])
        return np.array([0, 0, 0, 0, *dlam])

def loss_optimizer(lam, t, p, func):
    return np.sum((p-func(t, *lam))**2)/len(t)


"""
Plot
"""

def complex2idx(original, cplx):
    return np.array([np.where(np.isin(original, cplx[i])) for i in range(len(cplx))]).reshape(len(cplx))




def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)