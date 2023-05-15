import os
import time
import datetime
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy import optimize
from scipy.optimize import minimize, curve_fit
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from numpy.linalg import inv
from qiskit.algorithms.optimizers import ADAM

from preparation import grad_lam

"""
TODOS:
    -use lm fit to fit the 
"""


class Rabi_Fit():
    def __init__(self, t_max, dt, exp_data, rabi_models, params):
        """
        args:
            exp_data: dataframe of the experiment outcome
            UM_labels: list of all possible Us and Ms config
            rabi_models: list of functions
            params: list of parameter of the estimation before, shape of (6,)
            is_initial: is the probability initial, only true if the 
        """
        self.t_max = t_max
        self.dt = dt
        self.data = exp_data
        self.lam = None
        self.rabi_models = rabi_models
        self.params = params
        self.is_initial = np.zeros(len(rabi_models))

    def zero_padding(self, t, p, n_pad):
        t_max = np.max(t)
        t_min = np.min(t)
        n_t = len(t)
        del_t = t[1]-t[0]
        t_pad = np.linspace(t_min, t_max+n_pad*del_t, len(t)+n_pad)
        p_pad = np.pad(p, (0, n_pad))
        return t_pad, p_pad


    def inital_guess_freq(self, t, p, t_max, N_interp, N_pad=int(5e3)):
        t_intp = np.linspace(0, t_max, N_interp)[1:]#  !Important, endpoint= False
        p_intp = np.interp(t_intp, t, p)
        t_pad, p_pad = self.zero_padding(t_intp, p_intp, n_pad=N_pad)
        N_tot = len(t_pad)
        del_t = np.diff(t_pad)[0]
        yf_freq = fft(p_pad-np.mean(p_pad))
        yf = fft(p_pad-np.mean(p_pad))
        y_fft = 2.0/N_tot * np.abs(yf[:N_tot//2])
        y_fft_freq = 2.0/N_tot * np.abs(yf_freq[:N_tot//2])
        omgf = fftfreq(N_tot, del_t)[:N_tot//2]*np.pi*2/1e6
        omg_max = omgf[np.where(y_fft_freq==np.max(y_fft_freq[1:]))][0]
        amp = np.abs(yf[np.where(y_fft==np.max(y_fft[1:]))][0])
        phaseshift = np.angle(yf[np.where(y_fft==np.max(y_fft[1:]))][0])
        
        # plt.figure()
        # plt.plot(omgf, y_fft_freq)
        # plt.grid()
        # plt.show()
        print(omg_max/2)
        return omg_max/2
        

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
            four = np.array([four_o0_re, four_o0_im]).reshape((2,1))
            M = np.array([
                [np.cos(k)-np.cos(k)**2, np.sin(k)*np.cos(k)],
                [np.sin(k)-np.sin(k)*np.cos(k), np.sin(k)**2]
            ])
            M = np.sum(M, axis = 2)
            inv_M = np.linalg.inv(M)
            A = np.matmul(inv_M, four)[0][0]
            B = np.matmul(inv_M, four)[1][0]
        else:
            A = (np.sum(p)-len(t))/(np.sum(np.cos(k))-len(t))
            B = 0
        return A, B

    def rough_params(self, n_data):
        t, p= self.get_tp(n_data)
        t_max = self.t_max

        n_rabi = n_data%3
        mu = self.inital_guess_freq(t, p, t_max, len(t)+5)
        A, B = self.get_fourier_coefficient(t, p, 2*mu*1e6, n_rabi)
        return mu, A, B

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
        t,p = self.get_tp(n_data)
        return len(t) < 5
    
    def loss_optimizer(self, lam, t, p, func):
        return np.mean((p-func(t, lam))**2)

    def to_lam(self, lam_inter):
        lam_rough = np.zeros((8))
        # interate over two freqencies
        for i in range(2):
            n_max = (i+1)*3-1
            mu_mean = np.mean(lam_inter[:n_max,0])
            b_i = lam_inter[n_max, 1]*mu_mean**2
            a_i = np.sqrt(np.abs(mu_mean**2-b_i**2))
            phi_i = np.arcsin(-1*lam_inter[n_max-2, 2]*mu_mean/b_i)
            lam_rough[i*4] = mu_mean
            lam_rough[i*4+1] = a_i
            lam_rough[i*4+2] = b_i
            lam_rough[i*4+3] = phi_i
        return lam_rough
    
    def get_avg_freq(self, lam_inter):
        lam_rough = self.to_lam(lam_inter)
        return lam_rough[0], lam_rough[4]

    def rough_estimation(self, n_tot, lam_last):
        """
            Return:
                lam_rough: np.array with shape (8,)
        """
        lam_inter = np.zeros((n_tot, 3))
        for n_data in range(n_tot):
            lam_inter[n_data,:] = self.rough_params(n_data)
        if lam_last is None:
            lam_rough = self.to_lam(lam_inter)
        else:
            mu0, mu1 = self.get_avg_freq(lam_inter)
            lam_rough = np.insert(lam_last, [0,3], [mu0, mu1]) 
        return lam_rough 
                

        
    def fit_rabi(self, n_data, lam_rough, method ="L-BFGS-B", lr=1e-5):
        """
        Args:
            n_data: number of your data, in the order of (U0,t,X), (U0,t,Y), (U0,t,Z), (U1,t,X), (U1,t,Y), (U1,t,Z)
            method: "GD" or "curve" or "L-BFGS-B"
        """
        rabi_func = self.rabi_models[n_data]
        t, p = self.get_tp(n_data)

        if method == "GD":
            result = self.gradient_descent(
                gradient=self.gradient_MSE,
                parameters=lam_rough,
                fit_func=rabi_func,
                loss_func=self.loss_optimizer,
                learn_rate=lr,
                n_data=n_data
            )
            lam_n = result[0]
        elif method == "curve":
            result = curve_fit(rabi_func, t.to_numpy().astype(np.float64), p.to_numpy().astype(np.float64), p0=lam_rough.astype(np.float64))
            lam_n = result[0] 
        elif method == "L-BFGS-B":
            result = minimize(self.loss_optimizer, x0=lam_rough, args=(t, p, rabi_func), method="L-BFGS-B")
            lam_n = result.x 
            # print(result.fun)
        elif method == "COBYLA":
            result = minimize(self.loss_optimizer, x0=lam_rough, args=(t, p, rabi_func), method="COBYLA")
            lam_n = result.x 
        loss_min = self.loss_optimizer(lam_n, t, p, rabi_func)
        if math.isnan(loss_min):
            loss_min = 1
        return result, loss_min
    
    def fit_params(self, method="L-BFGS-B", lam_last=None, lr=1e-5):
        n_tot = len(self.rabi_models)
        # preparation for estimation
        lam_rough = self.rough_estimation(n_tot, lam_last)
        lam_fit = np.zeros((n_tot, 8))
        loss_fit = np.zeros(n_tot)
        for n_data in range(n_tot):
            if self.is_empty(n_data):
                self.is_initial[n_data] = -1
                lam_n = np.array([1e6,1,1,1])
                loss_min = 1
                result = -1
                lam_fit[n_data, :] = self.dim_adjust(lam_n, n_data)
                loss_fit[n_data] = loss_min
                print("No estimation for nr. {} due to the lackness of data".format(n_data))
            else:
                self.is_initial[n_data] = 0
                result, loss_min = self.fit_rabi(n_data, lam_rough=lam_rough[n_data//3*4:n_data//3*4+4], method=method)
                lam_fit[n_data, :] = self.dim_adjust(result.x, n_data)
                loss_fit[n_data] = loss_min
        return np.mean(lam_fit, axis=0)*2, loss_fit
        

    def adam_optimizer(self, p, func, t, is_xy=False, func2=None, t2=None, p2=None):
        if is_xy==True:
            def adam_loss(x0):
                return 0.5*np.mean((p-func(t, x0))**2)+0.5*np.mean((p2-func2(t2, x0))**2)
        def adam_loss(x0):
            return np.mean((p-func(t, x0))**2)
        return adam_loss
    
    def fit_func(self, n_rabi, alpha=0, beta=0):
        if n_rabi == 0:
            def rabi_i(t, x0):
                mu = x0[0]*1e6
                phi = x0[1]
                return alpha*np.cos(phi)*(1-np.cos(2*mu*t))+beta*np.sin(phi)*np.sin(2*mu*t)
        elif n_rabi == 1:
            def rabi_i(t, x0):
                mu = x0[0]*1e6
                phi = x0[1]
                return alpha*np.sin(phi)*(1-np.cos(2*mu*t))+beta*np.cos(phi)*np.sin(2*mu*t)
        else:
            def rabi_i(t, x0):
                mu = x0[0]*1e6
                b = x0[1]*1e6
                return (1-b**2/mu**2)+(b**2/mu**2)*np.cos(2*mu*t)
        return rabi_i
    
    def adam(self, lam_n, dmu=3, with_f=False):
        print("adam optimization occurs")
        print(lam_n)
        lam0 = lam_n[:4]
        lam1 = lam_n[4:]
        tx0, px0 = self.get_tp(0)
        ty0, py0 = self.get_tp(1)
        tz0, pz0 = self.get_tp(2)
        tx1, px1 = self.get_tp(3)
        ty1, py1 = self.get_tp(4)
        tz1, pz1 = self.get_tp(5)
        # rabi_x = self.rabi_models[0]
        # rabi_z = self.rabi_models[2]
        loss = np.ones(4)
        
        adam_opt = ADAM(noise_factor=1e-5, eps=1e-7)
        
        adam_z = self.fit_func(n_rabi=2)
        loss_z0 = self.adam_optimizer(pz0, adam_z, tz0)
        res_z0 = adam_opt.minimize(loss_z0, 
                                    x0 = np.array([lam0[0], 0.1]),
                                    bounds=[(lam0[0]*(1-dmu), 0), (lam0[0]*(1+dmu), lam0[0]*(1+dmu))])
        
        mu0 = res_z0.x[0]
        b0 = res_z0.x[1]
        loss[1] = res_z0.fun
        if abs(mu0) > abs(b0):
            a0 = np.sqrt(mu0**2-b0**2)
        else:
            a0 = lam0[1]
        
        alpha0 = -1*a0*b0/mu0**2
        beta0 = -1*b0/mu0
        
        adam_x = self.fit_func(n_rabi=0, alpha=alpha0, beta=beta0)
        adam_y = self.fit_func(n_rabi=1, alpha=alpha0, beta=beta0)
        loss_x0 = self.adam_optimizer(px0, adam_x, tx0, is_xy=True, func2=adam_y,t2=ty0, p2=py0)
        res_x0 = adam_opt.minimize(loss_x0, 
                                    x0 = np.array([mu0, 0]),
                                    bounds=[(mu0*(1-dmu), -np.pi), (mu0*(1+dmu),  np.pi)])
        phi0 = res_x0.x[1]
        loss[0] = res_x0.fun

        loss_z1 = self.adam_optimizer(pz1, adam_z, tz1)
        res_z1 = adam_opt.minimize(loss_z1, 
                                   x0 = np.array([lam1[0], 0.1]),
                                   bounds=[(lam1[0]*(1-dmu), 0), (lam1[0]*(1+dmu), lam1[0]*(1+dmu))])
        mu1 = res_z1.x[0]
        b1 = res_z1.x[1]
        loss[3] = res_z1.fun
        if abs(mu0) > abs(b0):
            a1 = np.sqrt(mu0**2-b1**2)
        else:
            a1 = a0 = lam1[1]
        
        alpha1 = -1*a1*b1/mu1**2
        beta1 = -1*b1/mu1
        
        adam_x = self.fit_func(n_rabi=0, alpha=alpha1, beta=beta1)
        adam_y = self.fit_func(n_rabi=1, alpha=alpha1, beta=beta1)
        loss_x1 = self.adam_optimizer(px1, adam_x, tx1, is_xy=True, func2=adam_y,t2=ty1, p2=py1)
        res_x1 = adam_opt.minimize(loss_x1, 
                                    x0 = np.array([mu1, 0]),
                                    bounds=[(mu1*(1-dmu), -np.pi), (mu1*(1+dmu), np.pi)])
        phi1 = res_x1.x[1]
        loss[2] = res_x1.fun
        
        lam = np.array([a0, b0, phi0, a1, b1, phi1])
        if with_f:
            return np.array([mu0, a0, b0, phi0, mu1, a1, b1, phi1]), loss
        else:
            return lam, loss
        
        
            




