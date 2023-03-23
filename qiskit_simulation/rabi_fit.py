import os
import time
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy import optimize
from scipy.optimize import minimize, curve_fit
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from numpy.linalg import inv

from preparation import grad_lam

"""
TODOS:
    -use lm fit to fit the 
"""


class Rabi_Fit():
    def __init__(self, t_max, dt, exp_data, rabi_models):
        """
        args:
            exp_data: dataframe of the experiment outcome
            UM_labels: list of all possible Us and Ms config
            rabi_models: list of functions
        """
        self.t_max = t_max
        self.dt = dt
        self.data = exp_data
        self.lam = None
        self.rabi_models = rabi_models

    def zero_padding(self, t, p, n_pad):
        t_max = np.max(t)
        t_min = np.min(t)
        n_t = len(t)
        del_t = t[1]-t[0]
        t_pad = np.linspace(t_min, t_max+n_pad*del_t, len(t)+n_pad)
        p_pad = np.pad(p, (0, n_pad))
        return t_pad, p_pad


    def inital_guess_freq(self, t, p, t_max, N_interp, N_pad=500):
        t_intp = np.linspace(0, t_max, N_interp)[1:]#  !Important, endpoint= False
        p_intp = np.interp(t_intp, t, p)
        t_pad, p_pad = self.zero_padding(t_intp, p_intp, n_pad=N_pad)
        N_tot = len(t_pad)
        del_t = np.diff(t_pad)[0]
        yf = fft(p_pad)
        y_fft = 2.0/N_tot * np.abs(yf[:N_tot//2])
        omgf = fftfreq(N_tot, del_t)[:N_tot//2]*np.pi*2
        omg_max = omgf[np.where(y_fft==np.max(y_fft[1:]))][0]
        amp = np.abs(yf[np.where(y_fft==np.max(y_fft[1:]))][0])
        phaseshift = np.angle(yf[np.where(y_fft==np.max(y_fft[1:]))][0])
        return omg_max, amp, phaseshift 
        

    def get_tp(self, n_data):
        data = self.data[n_data]
        dt = self.dt
        n_start = 1 #throw away some unphysical points

        t = data["t"][n_start:]*dt
        p = data["count"][n_start:]
        return t, p
    

    def get_fourier_coefficient(self, t, p, omg0, n_rabi):
        #fourier
        four_o0_re = np.real(np.sum(p*np.exp(-1j*omg0*t)))
        four_o0_im = -1*np.imag(np.sum(p*np.exp(-1j*omg0*t)))
        four_0 = np.sum(p)
        k = omg0*t
        if n_rabi!=2:
            four = np.array([four_o0_re, four_o0_im, four_0]).reshape((3,1))
            M = np.array([[np.cos(2*k)**2, np.cos(2*k)*np.sin(2*k), np.cos(2*k)],
                        [np.cos(2*k)*np.sin(2*k), np.sin(k)**2, np.sin(2*k)],
                        [np.cos(2*k), np.sin(2*k), np.ones(len(k))]])
            M = np.sum(M, axis = 2)
            inv_M = np.linalg.inv(M)
            A = np.matmul(inv_M, four)[0][0]
            B = np.matmul(inv_M, four)[1][0]
            C = np.matmul(inv_M, four)[2][0]
        else:
            A = (np.sum(p)-len(t))/(np.sum(2*np.sin(k)*np.cos(k))-len(t))
            B = 0
            C = 0
        return A, B, C

    def rough_params(self, n_data):
        t, p= self.get_tp(n_data)
        t_max = self.t_max

        n_rabi = n_data%3
        mu, amp, phaseshift = self.inital_guess_freq(t, p, t_max, len(t)+5)
        # A, B, C = self.get_fourier_coefficient(t, p, mu/2, n_rabi)
        return np.array([mu/2, 1, phaseshift, 0])


        
    def gradient_descent(self, gradient, parameters, fit_func, loss_func, learn_rate, n_data, n_iter=50, decay_rate=0.95):
        vector = parameters
        t, p_exp = self.get_tp(n_data)
        loss_hist = np.zeros(n_iter)
        diff_hist = []
        vec_hist = []
        for i in range(n_iter):
            p_pred = fit_func(t, *vector)
            diff = -learn_rate * gradient(t, p_exp, p_pred, vector, n_data)
            vector += diff
            learn_rate = learn_rate*decay_rate

            loss = loss_func(vector, t, p_exp, fit_func)
            loss_hist[i] = loss
            diff_hist.append(diff)
            vec_hist.append(vector)
        vec_hist = np.array(vec_hist)
        if len(vec_hist) > 1:
            vector = vec_hist[np.where(loss_hist == loss_hist.min())][0]
        return vector, diff_hist, loss_hist

    def gradient_MSE(self, t, p, p_model, lam, n_data):
        """
        use chain rule deviation to calculate loss function gradient
        """
        dx = grad_lam(t, *lam, n_data)
        return np.array([ np.sum(-2*(p-p_model)*dx_i) for dx_i in dx]) 
    
    def dim_adjust(self, lam, n_data):
        if n_data//3 == 0:
            return np.array([*lam, 0,0,0,0])
        elif n_data//3 == 1:
            return np.array([0,0,0,0, *lam])
    
    def is_empty(self, n_data):
        t, p = self.get_tp(n_data)
        return len(t) < 5
    
    def loss_optimizer(self, lam, t, p, func):
        return np.mean((p-func(t, *lam))**2)

    def fit_params(self, n_data, method="GD", lr=1e-5):
        """
        Args:
            n_data: number of your data, in the order of (U0,t,X), (U0,t,Y), (U0,t,Z), (U1,t,X), (U1,t,Y), (U1,t,Z)
            method: "GD" or "curve" or "L-BFGS-B"
        """
        if self.is_empty(n_data):
            lam_n = np.array([1,1,1,0])
            loss_min = 1
            result = -1
            print("No estimation for nr. {} due to the lackness of data".format(n_data))
        else:
            lam_n = self.rough_params(n_data)
            rabi_func = self.rabi_models[n_data]
            t, p = self.get_tp(n_data)

            if method == "GD":
                result = self.gradient_descent(
                    gradient=self.gradient_MSE,
                    parameters=lam_n,
                    fit_func=rabi_func,
                    loss_func=self.loss_optimizer,
                    learn_rate=lr,
                    n_data=n_data
                )
                lam_n = result[0]
            elif method == "curve":
                result = curve_fit(rabi_func, t.to_numpy().astype(np.float64), p.to_numpy().astype(np.float64), p0=lam_n.astype(np.float64))
                lam_n = result[0] 
            elif method == "L-BFGS-B":
                result = minimize(self.loss_optimizer, x0=lam_n, args=(t, p, rabi_func), method="L-BFGS-B")
                lam_n = result.x 
                # print(result.fun)
            elif method == "COBYLA":
                result = minimize(self.loss_optimizer, x0=lam_n, args=(t, p, rabi_func), method="COBYLA")
                lam_n = result.x 
            elif method == "hybrid":
                result = self.gradient_descent(
                    gradient=self.gradient_MSE,
                    parameters=lam_n,
                    fit_func=rabi_func,
                    loss_func=self.loss_optimizer,
                    learn_rate=lr,
                    n_data=n_data
                )
                result = minimize(self.loss_optimizer, x0=result[0], args=(t, p, rabi_func), method="L-BFGS-B")
                lam_n = result.x
            loss_min = self.loss_optimizer(lam_n, t, p, rabi_func)
        return self.dim_adjust(lam_n, n_data), loss_min, result





