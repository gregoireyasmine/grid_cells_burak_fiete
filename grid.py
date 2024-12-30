from tilings import random_theta_tiling, deterministic_theta_tiling
from solvers import euler_solve, rk45_solve
import os
import torch
import numpy

wd = os.getcwd()
MODEL_SAVEDIR = wd + '/models/'
if not os.path.exists(MODEL_SAVEDIR):
    os.mkdir(MODEL_SAVEDIR)

SIM_SAVEDIR = wd + '/sim_data/'
if not os.path.exists(SIM_SAVEDIR):
    os.mkdir(SIM_SAVEDIR)

SOLVERS = {'euler': euler_solve, 'rk45': rk45_solve}
        
class Grid():
    def __init__(self, options, loadW=True, saveW=True, save_sim=False, sim_name = None):
        self.options = options
        self.n = options.n        
        self.device = options.device

               
        if self.options.tiling=='random' : 
            self.theta_pref = random_theta_tiling(options.n)
        else :
            self.theta_pref = deterministic_theta_tiling(options.n)
        
        self.vec_pref = torch.stack([torch.cos(self.theta_pref.flatten()), torch.sin(self.theta_pref.flatten())], 1).to(self.device) # prefered direction unit vector

        self.W = self.build_W(load=loadW, save=saveW)
 
        self.save_sim = save_sim
        self.sim_name = sim_name
        
        if options.W_periodic : 
            self.A = 1
        else : 
            raise NotImplementedError('Aperiodic case not implemented yet !')
            # A = torch.tensor(( XX < R - dr ) * 1 + ( XX >= R - dr ) * np.exp( -a0 * (XX - R + dr)**2 / dr**2 )).to(device)

        
        
    def build_W(self, load=True, save=True):
        fname = MODEL_SAVEDIR + self.options._str('W') + ".pth"

        if load and os.path.exists(fname):
            print(f"Recovering pre-computed matrix at {fname}")
            W = torch.load(fname)
            return W.to(self.device)
        
        print('Building new matrix from scratch')
        neuron_indexes = torch.arange(0, self.n**2, 1).to(self.device)
        
        x_loc = neuron_indexes%self.n
        y_loc = neuron_indexes//self.n
        
        X = torch.stack([x_loc, y_loc], 1)
        X_diff = torch.abs(X[:, torch.newaxis, :] - X[torch.newaxis, :, :] - self.options.W_l * self.vec_pref[:, torch.newaxis, :]) # distance h, v entre tous les points + shift
    
        if self.options.W_periodic : 
            X_diff =  torch.minimum(X_diff, self.n-X_diff)

        xx = torch.sum(X_diff.reshape(-1, X.shape[-1])**2, -1)
        W = self.options.W_a*(torch.exp(-self.options.W_gamma*xx)-torch.exp(-self.options.W_beta*xx)).reshape(self.n**2, self.n**2).to(self.device)
 
        if save :
            torch.save(W.cpu(), fname)
        return W


    def B(self, v):
        return self.A * (1 + self.options.alpha * self.vec_pref@v)

    
    def derivative(self):
        
        def derS(t, s, v_t):
            input_ = self.W @ s + self.B(v_t(t))
            return (1/self.options.tau) * (-s + input_ * (input_ > 0))
        
        return derS

    def simulate(v, s_0=None):
        if s_0 is None:
            s_0 = 00.1*torch.rand(n*n)
        v = v.to(device).float()
        s_0 = s_0.to(device).float()
        
        solver = SOLVERS[self.options.solver]
        S = solver(self.derivative(), self.options.dt, v, s_0, device=self.options.device)

        if self.save_sim: 
            if self.sim_name is None :
                self.sim_name = self.options._str('all') + '.pth'
                
                torch.save(S, SIM_SAVEDIR + self.sim_name)
