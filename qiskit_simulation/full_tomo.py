import os
import time
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from qiskit.providers.fake_provider import FakeAthens

from hamiltonian_learning import Hamiltonian_Learning
from preparation import rabi_x, rabi_y, rabi_z, discrete_cmap

def makepath(path):
    now = datetime.datetime.now()
    now = now.strftime('%Y_%m_%d__%H_%M_%S')
    now = str(now)
    path_now = os.path.join(path, now)
        
    if not os.path.exists(path_now):
        os.makedirs(path_now)
    print("figs saved in: {}".format(path_now))
    return path_now


batch_size = 25
N_iter = 1



t_max = 10*1e-7
dt = 2.2222222222222221e-10
N_t = 100
sig = 30
# ts = (np.linspace(0, t_max, int(N_t))/dt).astype(int)
ts = (np.linspace((4*sig)*dt, t_max, N_t)/dt).astype(int)
ts = ts[1:]
Us = [0, 1]
Ms = [0, 1, 2]
f_rabi = [rabi_x, rabi_y, rabi_z, rabi_x, rabi_y, rabi_z]


path = "C:\\Arbeit\\MasterArbeit\\echo_cr_tomography\\FakeAthenSimulation\\experiment_data\\single\\c1t0_04amp_04sig\\"
fig_path = "C:\\Arbeit\\qiskit_simulation\\full_tomo_figs"
fig_path = makepath(fig_path)


lam_labels = np.array(["mu0", "a0", "b0", "phi0", "mu1", "a1", "b1", "phi1"])
rabi_labels = np.array(["p_x", "p_y", "p_z", "p_x", "p_y", "p_z"])
U_label= [r"$(U_0, X)$",r"$(U_0, Y)$",r"$(U_0, Z)$",r"$(U_1, X)$",r"$(U_1, Y)$",r"$(U_1, Z)$"]
fake_UM = np.arange(6)

algorithm = Hamiltonian_Learning(backend=FakeAthens(), ts= ts, Us=Us, Ms=Ms, n_batch=batch_size, f_rabi=f_rabi)
loss_hist = np.zeros((6, N_iter))

for n_i in range(N_iter):
    algorithm.run_fake_full_tomo(path =path)
    exp_list = algorithm.get_reformed_data()
    for i in range(len(exp_list)):
        fig = plt.figure()
        plt.plot(exp_list[i]["t"]*dt*1e9, exp_list[i]["count"], ".", label=rabi_labels[i])
        plt.xlabel("t(ns)")
        plt.ylabel(r"$p_{rabi}$")
        plt.title(r'Raw Data x = (U_{0:.0f},t,M_{1:.0f})'.format(i//3, i%3))
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(fig_path, "raw_rabi_U{0}_M{1}.jpg".format(i//3, i%3)))
        plt.close(fig)

    params = algorithm.fit_params(method="L-BFGS-B", with_f=True)

    # params = np.array(algorithm.fit_params(method="COBYLA"))


    print(params)
    for i in range(len(exp_list)):
        n_rabi = i%3
        t = exp_list[i]["t"]*dt
        p = exp_list[i]["count"]
        if i//3 == 0:
            lam = params[:4]
        if i//3 == 1:
            lam = params[4:]
        p_pred = f_rabi[n_rabi](ts*dt, lam)
        # if n_rabi == 0:
        fig = plt.figure()
        plt.plot(t*1e9, p , ".")
        plt.plot(ts*dt*1e9, p_pred , "-", label=rabi_labels[i])
        plt.xlabel("t(ns)")
        plt.ylabel(r"$p_{rabi}$")
        plt.title(r'Raw Data x = (U_{0:.0f},t,M_{1:.0f})'.format(i//3, i%3))
        plt.grid()
        plt.savefig(os.path.join(fig_path, "fit_rabi_U{0}_M{1}.jpg".format(i//3, i%3)))
        plt.close(fig)
        
    loss = algorithm.get_loss()
    loss_hist[:,n_i] = loss[0]
    

print(loss)
# print(np.mean(loss[0][:,0]))
print(np.mean(loss))

