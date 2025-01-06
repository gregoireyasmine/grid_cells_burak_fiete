from tqdm import tqdm
import torch
from scipy.integrate import solve_ivp

def euler_solve(derS, dt, v, s_0, device='cpu', silent=False):
    batch_size, num_steps, num_dir = v.shape  # Number of trajectories in the batch
    num_neurons = s_0.shape[-1]  # Number of neurons (n^2)
    S = torch.zeros((batch_size, num_steps, num_neurons)).to(device)
    S[:, 0, :] = s_0  

    v_t = lambda t: v[:, t]  
    
    if not silent: 
        print('Simulating with Euler method')
        iterator = tqdm(range(num_steps-1))
        
    else : 
        iterator = range(num_steps-1)

    for t in iterator:
        der = derS(t, S[:, t], v_t)
        S[:, t + 1] = S[:, t] + dt * der
    
    return S

def rk45_solve(derS, dt, v, s_0):
    raise NotImplementedError('Rk45 not implemented yet on gpu, try euler')
    
    """
    v = v.T
    t_array = np.arange(0, v.shape[1], 1)*dt
    v_interp = interp1d(t_array, v, axis=1, kind="linear", fill_value="extrapolate") # TODO : interpolation on GPU
    v_t = lambda t: v_interp(t)
    print('Solving using rk45')
    T = v.shape[1]*dt
    S = solve_ivp(derS, t_span=(0, T), y0=s_0, t_eval=t_array, args = (v_t,), method="RK45")
    return S['y'].T
    """