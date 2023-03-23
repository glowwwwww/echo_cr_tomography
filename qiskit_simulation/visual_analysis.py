import os
import time
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import imageio


fig_path = "C:\\Arbeit\\qiskit_simulation\\figs\\"
folder = "2023_03_21__15_16_02_APS"
n_iter = 20



rawfolder = "\\fitdata"
animatefolder = fig_path+folder+"\\animate\\"

im_ls = np.array(os.listdir(fig_path+folder))


im_fit = np.zeros((2,3, n_iter), dtype=object)
im_sample = np.zeros(n_iter, dtype=object)
im_q = np.zeros(n_iter, dtype=object)
im_dq = np.zeros(n_iter, dtype=object)
for l in range(n_iter):        
    for u in range(2):
        for m in range(3):
            keyword = "U{1}_M{2}_L{0}".format(l,u,m)           
            im_fit[u,m,l] = fig_path+folder+rawfolder+"\\fit_rabi_"+keyword+".jpg"
    im_sample[l] = fig_path+folder+"\\sample_q_L{}".format(l)+".jpg"
    im_q[l] = fig_path+folder+"\\p_q_L{}".format(l)+".jpg"
    im_dq[l] = fig_path+folder+"\\diff_q_L{}".format(l)+".jpg"



"""
Plotting
"""
if not os.path.exists(animatefolder):
        os.makedirs(animatefolder)

for U in range(2):
    for M in range(3):
        im_rabi = [imageio.imread(im_fit[U,M,i]) for i in range(n_iter)]
        gifname = "evo_U{0}_M{1}.gif".format(U,M)
        imageio.mimwrite(animatefolder+gifname, im_rabi, fps=2)

im_samples = [imageio.imread(im_sample[i]) for i in range(n_iter)]
gifname_samp = "sample_evolution.gif"
imageio.mimwrite(animatefolder+gifname_samp, im_samples, fps=2)

im_q = [imageio.imread(im_q[i]) for i in range(n_iter)]
gifname_q = "query_evolution.gif"
imageio.mimwrite(animatefolder+gifname_q, im_q, fps=2)

im_dq = [imageio.imread(im_dq[i]) for i in range(n_iter)]
gifname_dq = "diff_query_evolution.gif"
imageio.mimwrite(animatefolder+gifname_dq, im_dq, fps=2)