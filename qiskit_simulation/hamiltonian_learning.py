import itertools
import os
import time
import datetime
import numpy as np
import pandas as pd
from scipy.stats import rv_discrete
from matplotlib import pyplot as plt
from numpy.linalg import inv
from scipy.optimize import minimize
from qiskit.providers.fake_provider import FakeAthens


from variable_quantum_circuit import Variable_Quantum_Circuit
from rabi_fit import Rabi_Fit
from preparation import gradient_rabi, rabi_x,rabi_y, rabi_z

lam_labels = np.array(["mu0", "a0", "b0", "phi0", "mu1", "a1", "b1", "phi1"])


class Hamiltonian_Learning():
    def __init__(self, backend, ts, Us, Ms, n_batch, f_rabi):
        """
        Init static variables and basic dimensions
        Args:
            ts: 1D numpy array with size (Nt) and type int
            Us: 1D numpy array with size of 2, set the state of the control qubit
            Ms: 1D numpy array with size of 3, set the effective measurement basis
            f_rabi: list of length 6, theory model for rabi oszillation with different query
        """
        self.backend = backend
        self.n_batch = n_batch
        self.f_rabi = f_rabi
        self.exp_data = None
        self.params = []
        self.params_n = np.zeros(8)
        self.adam_params = np.zeros(6)
        self.loss = []
        self.loss_n = np.zeros(6)
        self.n_iter = -1
        self.is_all_fit = False
        self._init_query_dim(ts, Us, Ms)
        self._init_q(ts, Us, Ms)
        self._init_backend_params(backend)


    """
    Initalization & Props def.
    """
    def _init_query_dim(self, ts, Us, Ms):
        self.dim_U = len(Us)
        self.dim_M = len(Ms)
        self.dim_t = len(ts)
        self.dim_UM = self.dim_U*self.dim_M
        self.dim_q = self.dim_UM*self.dim_t
        self.dim_lam = 8

    def _init_q(self, ts, Us, Ms):
        # q in the format of (t, U, M)    
        """
        Todos: better version of p_q 
        """
        d_UM = self.dim_UM
        d_t = self.dim_t

        UM_complex = (np.meshgrid(Us, Ms)[0]+1j*np.meshgrid(Us ,Ms)[1]).T.reshape(d_UM,)
        p_tUM = np.ones((d_UM, d_t))
        p_tUM = p_tUM/(d_UM*d_t)

        q = np.zeros((d_UM, d_t, 3), dtype=object)
        p_q = np.zeros((d_UM, d_t), dtype=float)

        for i in range(d_UM):
            for j in range(d_t):
                q[i,j,0] = ts[j]
                q[i,j,1] = UM_complex[i].real
                q[i,j,2] = UM_complex[i].imag
                p_q[i,j] = p_tUM[i,j]

        self.t_max = ts[-1]
        self.UM_labels = UM_complex
        self.q = q
        self.set_p_q(p_q)
        self.set_p_q_update(np.zeros((d_UM,d_t), dtype=float))

    def _init_backend_params(self, backend):
        self.dt = backend.configuration().dt
        

    def set_p_q(self, p_q):
        if np.sum(p_q) < 0.9:
            print("sum of the probability should be 1")
        elif np.sum(p_q) < 1:
            correction = (1-np.sum(p_q))/p_q.size
            p_q = p_q + correction
        self.p_q = p_q
    
    def set_params_n(self, params_n):
        self.params_n = params_n 
        
    def set_params(self, params):
        self.params = params    
          
    def add_loss(self, loss):
        self.loss.append(loss)
            
    def set_loss_n(self, loss_n):
        self.loss_n = loss_n
        
    def set_n_iter(self, n_iter):
        self.n_iter = n_iter
        
    def get_loss_n(self):
        return self.loss_n
        
    def set_q_sample(self, q_sample):
        self.q_sample = q_sample

    def set_adam(self, adams_par):
        self.adam_params = adams_par

    def get_p_q(self):
        return self.p_q
    
    def get_params(self):
        return self.params
    
    def get_params_n(self):
        return self.params_n
    
    def get_data(self):
        return self.exp_data
    
    def get_new_data(self):
        data = self.get_data()
        n_iter = self.get_n_iter()
        return data.loc[data["n_iter"]==n_iter].reset_index(drop=True)

    def get_hist_data(self):
        data = self.get_data()
        n_iter = self.get_n_iter()
        if n_iter > 0:
            return data.loc[data["n_iter"]!=n_iter].reset_index(drop=True)
        else:
            return pd.DataFrame()
    
    def get_reformed_data(self):
        data = self.get_data()
        UM_labels = self.UM_labels
        return self._reform_data(data, UM_labels)

    def get_n_data(self, UM):
        return np.where(self.UM_labels == UM)[0][0]

    def get_n_iter(self):
        return self.n_iter

    def get_q_sample(self):
        return self.q_sample
    
    def get_loss(self):
        return self.loss
    
    def get_p_q_update(self):
        return self.p_q_update
    
    def get_adam(self):
        return self.adam_params
    
    def _update_params(self, params, n):
        params_n = self.get_params_n()
        params_n[n,:] = params
        self.set_params_n(params_n)
        if n == 5:
            params_hist = self.get_params()
            params_hist.append(params_n)
            self.set_params(params_hist)
            self.set_params_n(np.zeros((6,8)))
    

    def _update_loss(self, loss, n):
        loss_n = self.get_loss_n()
        loss_n[n] = loss
        self.set_loss_n(loss_n)
        if n == 5:
            print("update loss")
            loss_hist = self.get_loss()
            loss_hist.append(loss_n)
            self.set_loss(loss_hist)
            self.set_loss_n(np.zeros(6))
            
    def _update_adam(self, adam_result):
        self.adam_params = adam_result
        

    def set_data(self, data):
        self.exp_data = data


    def set_weight(self, weight):
        n_batch = self.n_batch
        data_hist = self.get_hist_data()
        data_new = self.get_new_data()
        data_new.loc[:,"weight"] = weight
        data = pd.concat([data_hist, data_new])
        self.set_data(data)

    def set_p_q_update(self, p_q_update):
        self.p_q_update = p_q_update
        
    """
    Experiment Preparation
    """

    def _accept_rej(self):
        """
        sample n_batch from distribution q
        return x_sample
        """
        f = self.p_q.reshape((self.dim_q,))
        x = self.q.reshape((self.dim_q, 3))
        N = self.n_batch

        idx = np.arange(len(x))
        q_samp = rv_discrete(name='q_samp', values=(idx, f))
        sample_idx = q_samp.rvs(size=N)
        return x[sample_idx]

    def _set_iter_exp_data(self):
        q_sample = self.get_q_sample()
        n_iter = self.get_n_iter()
        data = pd.DataFrame()
        data["t"] = q_sample[:,0].astype(int)
        data["U"] = q_sample[:,1].astype(int)
        data["M"] = q_sample[:,2].astype(int)
        data["UM"] = data["U"]+1j*data["M"]
        data["count"] = 0
        data["weight"] = 1
        data["n_iter"] = n_iter
        for labels in lam_labels:
            data[labels] = 0
        if n_iter > 0:
            data_hist = self.get_hist_data()
            data = pd.concat([data_hist, data])
        self.set_data(data)
    
    def get_n_data_list(self):
        data = self.get_data()
        n_data_list = np.array([self.get_n_data(data["UM"].iloc[i]) for i in range(data.shape[0])])
        return n_data_list


    def set_lam_to_data(self, lam):
        data = self.get_data()
        for i in range(self.dim_lam):
            data[lam_labels[i]] = lam[i]
        self.set_data(data)



    """
    Data Post processing in Experiment
    """
    
    def _mean_duplicate_data(self, data, coldup=["t", "UM"], colmean=["count"]):
        dupitems = data.loc[data[coldup].duplicated(keep=False)]
        mean_counts = dupitems.groupby(coldup)[colmean].mean()
        for grouplabel, items in mean_counts.iterrows():
            dupitems.loc[(dupitems["t"]==grouplabel[0])&(dupitems["UM"]==grouplabel[1]), colmean] = items["count"]
        restitems = data.loc[np.invert(data[coldup].duplicated(keep=False))]
        return pd.concat([dupitems, restitems]).reset_index(drop=True)
        

    def _reform_data(self, exp_data, UM_labels):
        """
        return data in list sorted by t, p_rabi in the order of UM_labels
        """
        return [exp_data.loc[exp_data["UM"]==UM_labels[i]].sort_values(by="t") for i in range(len(UM_labels))]

    def inv_weight_mat(self, weight, FI_q):
        weighted_mat = np.sum([abs(weight[i])*FI_q[:,:,i] for i in range(FI_q.shape[-1])], axis=0)
        return np.trace(inv(weighted_mat))

    def _query_fisher_info(self, df):
        n_data = self.get_n_data(df["UM"])
        lam = np.real(df[lam_labels].to_numpy())
        dx_i = gradient_rabi(np.real(df["t"]*self.dt), *lam, n_data)
        fi_x = np.zeros((self.dim_lam, self.dim_lam))
        for i in range(self.dim_lam):
            for j in range(self.dim_lam):
                fi_x[i,j] = dx_i[i]*dx_i[j]
        return fi_x

    def _batch_fisher_info(self):
        data = self.get_new_data()
        fi_q = np.zeros((self.dim_lam, self.dim_lam, self.n_batch))
        for i, data_row in data.iterrows():
            fi_x = self._query_fisher_info(data_row)
            fi_q[:,:,i] = fi_x
        return fi_q
    
        
    def _minimize_fisher_info(self):
        fi_q = self._batch_fisher_info()
        initial_weight = self.get_new_data()["weight"]
        res = minimize(self.inv_weight_mat, x0=initial_weight/len(initial_weight), args=fi_q, method="COBYLA")
        w_opt = np.abs(res.x+np.min(res.x))
        w_opt = w_opt/np.sum(w_opt)
        return w_opt
    

    def mix_p_q(self, lr):
        n_batch = self.n_batch
        n_total = self.dim_q
        p_q = self.p_q
        q = self.q
        data = self.get_new_data()

        pu = n_batch/n_total*lr
        p_q_update = np.zeros_like(p_q)
        
        for i, row in data.iterrows():    
            idx_i = self.get_n_data(row["UM"])
            idx_j = np.where(q[0,:,0] == row["t"])[0][0]
            p_q_update[idx_i,idx_j] = np.real(row["weight"])

        self.set_p_q_update(p_q_update)

        # p_q_update = p_q_update-np.mean(p_q_update)
        p_q = (1-pu)*p_q+p_q_update*pu
        if np.sum(p_q)<1:
            p_q = p_q/np.sum(p_q)
        
        self.set_p_q(p_q)
        

    """
    Accessable Methods
    """
    def add_fake_noise(self, data, var):
        noise = np.random.normal(0, var, len(data))
        return data+noise
    
    def read_df(self, path, dataname):
        df = pd.read_csv(path+dataname, index_col=0)
        float_cols = df.columns.to_list()
        float_cols.remove("UM")
        float_cols.remove("count")

        df[float_cols] = df[float_cols].astype(int)
        df["UM"] = df["UM"].astype(np.complex128)
        df["count"] = df["count"].astype(float)
        df["count"] = self.add_fake_noise(df["count"], var=0*np.mean(np.abs(df["count"])))
        return df

    def read_data(self, path):
        data = []
        UM_cplx = self.UM_labels
        for i in range(self.dim_UM):
            dataname = "full_U{}_M{}.csv".format(int(UM_cplx[i].real), int(UM_cplx[i].imag))
            data.append(self.read_df(path, dataname))
        return pd.concat(data).reset_index(drop=True)
        
        
    def sample_query(self):
        n_iter = self.get_n_iter()
        q_sample = self._accept_rej()
        self.set_q_sample(q_sample)
        self._set_iter_exp_data()
        
    def run_fake_full_tomo(self, path):
        full_data = self.read_data(path)
        full_data = full_data.drop(full_data[full_data["t"] == 0].index)
        full_data["weight"] = 1
        full_data["n_iter"] = -1
        for labels in lam_labels:
            full_data[labels] = 0
        self.set_data(full_data)

    def run_fake_sim(self, path):
        n_iter = self.get_n_iter()
        data_new = self.get_new_data()
        data_hist = self.get_hist_data()
        full_data = self.read_data(path)
        
        result = np.zeros((self.n_batch))
        for i, row in data_new.iterrows():
            result[i] = full_data.loc[(full_data["UM"]==row["UM"])&(full_data["t"]==int(row["t"].real))]["count"]
        data_new.loc[:,"count"] = result
        data_new = self._mean_duplicate_data(data_new)
        data = pd.concat([data_hist, data_new])
        data = self._mean_duplicate_data(data)
        self.set_data(data)

    
    def run_simulation(self, cbit, tbit):
        """
        Todos: run only the new simulations
        """
        q_sample = self.get_q_sample()
        data = self.get_data()
        n_iter = self.get_n_iter()
        
        result = np.zeros((self.n_batch))
        q_exp = Variable_Quantum_Circuit(self.backend, cbit, tbit, amp=1+0j, rise_ratio=0.1, num_shots=1024)

        for i in range(self.n_batch):
            result[i] = q_exp.run(*q_sample[i])
        data["count"] = result
        is_dup = np.sum(data[["t","UM"]].duplicated())
        data_filtered = self.mean_duplicate(data, n_iter, ["t","UM"], is_dup)
        self.exp_data = self.set_data(data_filtered)
        

    def fit_params(self, method, with_f=False):
        dt = self.dt
        data = self.get_data()
        adam_pars = self.get_adam()    
        UM_labels = self.UM_labels
        f_rabi = self.f_rabi
        t_max = self.t_max*dt
        n_iter =  self.get_n_iter()
        
        if n_iter>0:
            params0 = np.delete(adam_pars, [0,4])
        else:
            params0 = None
        
        sorted_data = self._reform_data(data, UM_labels)
        
        theory_model = Rabi_Fit(t_max, dt, sorted_data, f_rabi, adam_pars)
        prams, loss = theory_model.fit_params(method, params0)
        if np.sum(theory_model.is_initial)==0:
            adam_results, loss = theory_model.adam(prams, with_f=with_f)
            self.set_adam(adam_results)
        else:
            adam_results = np.array([1, 1, 1, 0, 1, 1, 1, 0])
            loss = np.delete(loss, [1,5])
            self.set_adam(adam_results)
            print(loss)
        self.add_loss(loss)
        self.set_lam_to_data(adam_results)
        return adam_results


    def optimize_query(self, lr):
        w_opt = self._minimize_fisher_info()
        self.set_weight(w_opt)
        self.mix_p_q(lr)
    
    def run(self, path, n_iter_max):
        for n_i in range(n_iter_max):
            print("running the hamiltonian learning algorithmus")
            self.set_n_iter(n_i)
            self.sample_query()
            self.run_fake_sim(path)
            self.fit_params()
            self.optimize_query()
            


"""
Testing Programm
"""
# t_max = 20*1e-7
# dt = 2.2222222222222221e-10
# N_t = 100
# ts = (np.linspace(0, t_max, N_t)/dt).astype(int)
# Us = [0, 1]
# Ms = [0, 1, 2]
# f_rabi = [rabi_xy, rabi_xy, rabi_z, rabi_xy, rabi_xy, rabi_z]
# path = "C:\Physics_Master\Master_Arbeit\HAL_demo\hal_simulation\FakeAthenSimulation\\full_to_rec\\"

# algorithm = Active_Learner(backend=FakeAthens(), ts= ts, Us=Us, Ms=Ms, n_batch=30, f_rabi=f_rabi)
# algorithm.sample_query()
# algorithm.run_fake_sim(path =path)
# algorithm.fit_params()
# algorithm.optimize_query()