from tqdm import tqdm

def euler_solve(derS, dt, v, s_0, device='cpu', silent=False):
    S = torch.zeros((len(v), s_0.shape[0], s_0.shape[1])).to(device)
    S[0] = s_0
    v_t = lambda t: v[t]
    
    if not silent : 
        print('Simulating with euler method')

    if silent : 
        iterator = range(len(v[:-1]))
    else : 
        iterator = tqdm(range(len(v[:-1])))
    for t in iterator:
        S[t + 1] = S[t] + dt*derS(t, S[t], v_t)
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