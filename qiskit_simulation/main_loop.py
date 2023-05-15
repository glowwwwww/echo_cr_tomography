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

def makefoler(path, folder):
    complete_path = os.path.join(path, folder)
    if not os.path.exists(complete_path):
        os.makedirs(complete_path)
    return complete_path

iter_max = 8
batch_size = 30


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
fig_path = "C:\\Arbeit\\MasterArbeit\\echo_cr_tomography\\qiskit_simulation\\adam_figs"
fig_path = makepath(fig_path)
raw_path = makefoler(fig_path, "rawdata")
fit_path = makefoler(fig_path, "fitdata")


lam_labels = np.array(["mu0", "a0", "b0", "phi0", "mu1", "a1", "b1", "phi1"])
rabi_labels = np.array(["p_x", "p_y", "p_z", "p_x", "p_y", "p_z"])
U_label= [r"$(U_0, X)$",r"$(U_0, Y)$",r"$(U_0, Z)$",r"$(U_1, X)$",r"$(U_1, Y)$",r"$(U_1, Z)$"]
fake_UM = np.arange(6)



algorithm = Hamiltonian_Learning(backend=FakeAthens(), ts= ts, Us=Us, Ms=Ms, n_batch=batch_size, f_rabi=f_rabi)
loss_hist = []
for n_iter in range(iter_max):
    print("Iteration Nr {}\n".format(n_iter))
    algorithm.set_n_iter(n_iter)
    algorithm.sample_query()
    p_q = algorithm.get_p_q()
    
    fig = plt.figure()
    plt.rcParams.update({'font.size': 14})
    plt.pcolormesh(ts*dt*1e9, fake_UM, p_q*100, norm=colors.PowerNorm(gamma=0.5), cmap="Blues")
    cbar = plt.colorbar()
    cbar.set_label(r'$p(q_i)$ (%)')
    plt.yticks(fake_UM, labels=U_label)
    plt.ylabel('U')
    plt.xlabel('t(ns)')
    plt.title(r'Probability Distribution p(q), $n_{{iter}}=${}'.format(n_iter))
    plt.tight_layout()
    # plt.axis.label.set_size(16)
    plt.savefig(os.path.join(fig_path, "p_q_L{0}.jpg".format(n_iter)))
    plt.close(fig)


    data = algorithm.get_data()
    batch_size = algorithm.n_batch
    UM_complex = algorithm.UM_labels
    N_UM = len(UM_complex)


    
    UMidx = algorithm.get_n_data_list()
    H, xedges, yedges = np.histogram2d(data["t"]*dt*1e9, UMidx, bins=[N_t, N_UM])
    H = H.T

    binwidth = (N_UM-1)/N_UM/2
    fake_UM = np.linspace(1, N_UM*2-1, N_UM)*binwidth

    fig = plt.figure()
    colormesh = plt.pcolormesh(xedges, yedges, H, cmap=discrete_cmap(int(np.max(H)+1), "Blues"))
    plt.colorbar(colormesh, ticks=range(int(np.max(H)+1)))
    plt.clim(-0.5, int(np.max(H)+1) - 0.5)
    plt.yticks(fake_UM, labels=U_label)
    plt.ylabel('U')
    plt.xlabel('t (ns)')
    plt.title(r'2D Histogramm Sampling Query with $N_b$={}'.format(batch_size))
    plt.savefig(os.path.join(fig_path, "sample_q_L{}.jpg".format(n_iter)))
    plt.close(fig)
    
    algorithm.run_fake_sim(path =path)
    exp_list = algorithm.get_reformed_data()
    for i in range(len(exp_list)):
        fig = plt.figure()
        plt.plot(exp_list[i]["t"]*dt*1e9, exp_list[i]["count"], ".", label=rabi_labels[i])
        plt.xlabel("t(ns)")
        plt.ylabel(r"$p_{rabi}$")
        plt.title(r'Raw Data x = (U_{0:.0f},t,M_{1:.0f}),  $n_{{iter}}$={2}'.format(UM_complex[i].real, UM_complex[i].imag, n_iter))
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(raw_path, "raw_rabi_U{1}_M{2}_L{0}.jpg".format(n_iter, i//3, i%3)))
        plt.close(fig)
    
    params = np.array(algorithm.fit_params(method="L-BFGS-B", with_f=True))
    # params = np.array(algorithm.fit_params(method="curve"))
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
        plt.title(r'Raw Data x = (U_{0:.0f},t,M_{1:.0f}) $n_{{iter}}$={2}'.format(UM_complex[i].real, UM_complex[i].imag, n_iter))
        plt.grid()
        plt.savefig(os.path.join(fit_path, "fit_rabi_U{1}_M{2}_L{0}.jpg".format(n_iter, i//3, n_rabi)))
        plt.close(fig)
    
    # loss_hist.append(loss)

    algorithm.optimize_query(lr=1)
    
    p_q_update = algorithm.get_p_q_update()
    
    fig = plt.figure()
    plt.pcolormesh(ts*dt*1e9, fake_UM, p_q_update*100, norm=colors.PowerNorm(gamma=0.5), cmap="Blues")
    cbar = plt.colorbar()
    cbar.set_label(r'$p(q_i)$ (%)')
    plt.yticks(fake_UM, labels=U_label)
    plt.ylabel('U')
    plt.xlabel('t(ns)')
    plt.title(r'Learned probability p(q), $n_{{iter}}=${}'.format(n_iter))
    plt.savefig(os.path.join(fig_path, "diff_q_L{0}.jpg".format(n_iter)))
    plt.close(fig)
    
print(algorithm.get_adam())
loss_hist = np.array(algorithm.get_loss())

mean_loss = np.array([np.mean(loss_hist[i]) for i in range(len(loss_hist))])    
sig_loss  = np.array([np.std(loss_hist[i], ddof=1) for i in range(len(loss_hist))])
nr_data = np.arange(iter_max).astype(int)


fig = plt.figure()
plt.plot(nr_data, loss_hist, ".-")
# plt.yscale("log")
# plt.fill_between(nr_data, mean_loss - sig_loss, mean_loss + sig_loss, alpha=0.2)
plt.title("Loss Evolution during Learning")
plt.grid()
plt.savefig(os.path.join(fig_path, "loss_hist.jpg"))
plt.close(fig)


fig = plt.figure()
plt.plot(nr_data, loss_hist, ".-")
plt.yscale("log")
plt.grid()
# plt.fill_between(nr_data, mean_loss - sig_loss, mean_loss + sig_loss, alpha=0.2)
plt.title("ln(Loss) Evolution during Learning")
plt.savefig(os.path.join(fig_path, "log_loss_hist.jpg"))
plt.close(fig)
print(mean_loss)
# print(loss_hist)

# full_loss = 0.004988609002410701
# full_loss_0_3 = 0.02839245117263177
# full_loss_0_1 = 0.007424841133476666
# full_loss_0_01= 0.005003211876139741
# n_total = 600/batch_size
fig = plt.figure()
plt.plot(nr_data, mean_loss, ".-", label="active learning")
# plt.hlines(full_loss, 0, n_iter, linestyles="dashed", label="full tomography")
plt.yscale("log")
# plt.xscale("log")
plt.grid()
plt.xlabel(r"$n_{{iter}}$")
plt.fill_between(nr_data, mean_loss - sig_loss/2, mean_loss + sig_loss/2, alpha=0.2, label=r"$1-\sigma$ interval")
plt.legend()
plt.title("Mean ln(Loss) Evolution during Learning")
plt.savefig(os.path.join(fig_path, "mean_log.jpg"))
plt.close(fig)
print(mean_loss)
# print(loss_hist)
