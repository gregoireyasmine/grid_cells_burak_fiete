from tilings import random_theta_tiling, deterministic_theta_tiling
from solvers import euler_solve, rk45_solve
import os
import torch
import numpy
from time import time

wd = os.getcwd()
MODEL_SAVEDIR = wd + '/models/'
if not os.path.exists(MODEL_SAVEDIR):
    os.mkdir(MODEL_SAVEDIR)

SIM_SAVEDIR = wd + '/sim_data/'
if not os.path.exists(SIM_SAVEDIR):
    os.mkdir(SIM_SAVEDIR)

SOLVERS = {'euler': euler_solve, 'rk45': rk45_solve}
        
class Grid():
    """Burak & Fiete (2009) grid cells continuous attractor model"""
    
    def __init__(self, options, loadW=True, saveW=True, save_sim=False, initialize=True, load_s0 = True, save_s0 = True):
        self.options = options
        self.n = options.n        
        self.device = options.device

               
        if self.options.tiling=='random' : 
            self.theta_pref = random_theta_tiling(options.n)
        else :
            self.theta_pref = deterministic_theta_tiling(options.n)
        
        self.vec_pref = torch.stack([torch.cos(self.theta_pref.flatten()), torch.sin(self.theta_pref.flatten())], 1).to(self.device) # prefered direction unit vector

        self.build_W(load=loadW, save=saveW)
 
        self.save_sim = save_sim
        self.grid_id = self.options._str('all')
        
        if options.W_periodic : 
            self.A = 1
        else : 
            raise NotImplementedError('Aperiodic case not implemented yet !')
            # A = torch.tensor(( XX < R - dr ) * 1 + ( XX >= R - dr ) * np.exp( -a0 * (XX - R + dr)**2 / dr**2 )).to(device)

        self.s0 = 00.1*torch.rand(self.n**2)
        
        if initialize:
            self.heal(restore = load_s0)

        
        
    def build_W(self, load=True, save=True):
        fname = MODEL_SAVEDIR + "Wmat_" +self.options._str('W') + ".pth"

        if load and os.path.exists(fname):
            print(f"Recovering pre-computed matrix found at {fname}")
            W = torch.load(fname, weights_only=True)
            self.W = W.to(self.device)
            return 
            
        print('Building new matrix from scratch')
        neuron_indexes = torch.arange(0, self.n**2, 1).to(self.device)
        
        x_loc = neuron_indexes%self.n
        y_loc = neuron_indexes//self.n
        
        X = torch.stack([x_loc, y_loc], 1)
        X_diff = torch.abs(X[:, torch.newaxis, :] - X[torch.newaxis, :, :] - self.options.W_l * self.vec_pref[:, torch.newaxis, :]) # distance h, v entre tous les points + shift
    
        if self.options.W_periodic : 
            X_diff =  torch.minimum(X_diff, self.n-X_diff)

        xx = torch.sum(X_diff.reshape(-1, X.shape[-1])**2, -1)
        self.W = self.options.W_a*(torch.exp(-self.options.W_gamma*xx)-torch.exp(-self.options.W_beta*xx)).reshape(self.n**2, self.n**2).to(self.device)
 
        if save :
            torch.save(self.W.cpu(), fname)


    def B(self, v):
        return self.A * (1 + self.options.alpha * torch.matmul(v, self.vec_pref.T))

    
    def derivative(self):
        
        def derS(t, s, v_t):
            input_ = torch.matmul(s, self.W) + self.B(v_t(t))
            return (1/self.options.tau) * (-s + input_ * (input_ > 0))
        
        return derS
    
    
    def simulate(self, v, sim_id=None, s_0=None, update_s0=False, silent=False, load=False):
        if sim_id is None:
            sim_id = self.grid_id + '_' + str(int(time())) + '.pth'
    
        path = SIM_SAVEDIR + sim_id
        
        if load and os.path.exists(path):
            print(f"loading pre-computed trajectory at {path}")
            S = torch.load(path, weights_only=True)
            return S
        
        if s_0 is None:
            s_0 = self.s0.clone()
    
        if type(v) != torch.Tensor:
            v = torch.Tensor(v)
        if type(s_0) != torch.Tensor:
            s_0 = torch.Tensor(s_0)
    
        if v.device != self.device:
            v = v.to(self.device).float()  
        if s_0.device != self.device:
            s_0 = s_0.to(self.device).float() 
    
        if v.dim() == 2:  # Handle batches
            v = v.unsqueeze(0)

        batch_size = v.shape[0]
        if s_0.dim() == 1:  
            s_0 = s_0.expand(batch_size, -1)  # Expand initial state to match the batch size
        elif s_0.dim() == 2 and s_0.shape[0] != batch_size:  # Mismatched batch size
            raise ValueError("Batch size of v and s_0 must match.")

        solver = SOLVERS[self.options.solver]
    
        S = solver(self.derivative(), self.options.dt, v, s_0, device=self.options.device, silent=silent)
    
        if self.save_sim:
            print(f'saving simulation at {path}')
            torch.save(S.cpu(), path)
    
        if update_s0:
            self.s0 = S[0, -1]
    
        return S.cpu()

    
    def heal(self, restore = True, save=True, T=0.25, v_norm=0.8, angles = torch.tensor([0, torch.pi/5, torch.pi/2, -torch.pi/5])):
        fname = MODEL_SAVEDIR + "s0_" + self.grid_id + ".pth"
        
        if restore and os.path.exists(fname):
            print(f"Restoring pre-computed initial state found at {fname}")
            self.s0 = torch.load(fname, weights_only=True)
            return
            
        print("Building init state from scratch, might take a few seconds")

        num_steps = int(T/self.options.dt)
        v = torch.zeros((num_steps, 2))
        
        self.simulate(v, update_s0 = True, silent=True)
        
        for a in angles:
            v = v_norm * torch.tensor([torch.cos(a), torch.sin(a)]) * torch.ones_like(v)
            traces = self.simulate(v, update_s0=True, silent=True)
        
        v = torch.zeros((num_steps, 2))
        self.simulate(v, update_s0=True, silent=True)

        if save :
            torch.save(self.s0.cpu(), fname)
            
            


