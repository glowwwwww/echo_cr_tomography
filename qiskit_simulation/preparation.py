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


U_label= [r"$(U_0, X)$",r"$(U_0, Y)$",r"$(U_0, Z)$",r"$(U_1, X)$",r"$(U_1, Y)$",r"$(U_1, Z)$"]

def rabi_xy(t, mu, A, B, C):
    return A*np.cos(2*mu*t)+B*np.sin(2*mu*t)-A


def rabi_z(t, mu, A, B, C):
    return A*np.cos(2*mu*t)+(1-A)

def grad_lam(t, mu, A, B, C, n_rabi):
    if n_rabi !=2:
        dmu = -2*A*t*np.sin(2*mu*t)+2*B*t*np.cos(2*mu*t)
        dA = np.cos(2*mu*t)-1
        dB = np.sin(2*mu*t)
        dC = 0
    else:
        dmu = -2*A*t*np.sin(2*mu*t)
        dA = np.cos(2*mu*t)-1
        dB = 0
        dC = 0
    return dmu, dA, dB, dC

def gradient_rabi(t, mu0, A0, B0, C0, mu1, A1, B1, C1, n_data):
    n_rabi = n_data%3
    if n_data//3==0:
        dmu, dA, dB, dC = grad_lam(t, mu0, A0, B0, C0, n_rabi)
        return np.array([dmu, dA, dB, dC, 0, 0, 0, 0])
    elif n_data//3==1:
        dmu, dA, dB, dC = grad_lam(t, mu1, A1, B1, C1, n_rabi)
        return np.array([0, 0, 0, 0, dmu, dA, dB, dC])


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