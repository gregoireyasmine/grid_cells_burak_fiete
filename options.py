class Options():
    def __init__(self, lambda_net = 15, e = 1.15):
        self.lambda_net = lambda_net # approximate number of neurons between two blob centers
        self.e = e # controls the distance between inner limit and upper limit of the inhibitory surround, recommend 1.15 for n=32, 1.05 for n=128

        self.tau = 1E-2 # time constant of the network
        self.alpha = 0.2 # strength of the speed input
        self.W_gamma = self.e*3/(self.lambda_net**2) # outer limit of the inhib surround
        self.W_beta = 3/(self.lambda_net**2) # inner limit of the inhib surround
        self.W_periodic = True
        self.W_a = 1.5 # if a > 1, close surround of cells is activatory
        self.W_l = 2 # spatial shift size of the center of the inhibitory surround
        self.n=40 
        self.solver = 'euler'
        self.device = 'cpu'
        self.dt = 1E-4
        self.tiling = 'deterministic'


    def W_options(self):
        return {'beta' : self.W_beta,
                'gamma': self.W_gamma,
                'periodic': self.W_periodic,
                'a' : self.W_a,
                'l' : self.W_l,
                'n': self.n}


    def dynamics_options(self):
        return {'tau': self.tau, 'alpha' : self.alpha, 'tiling': self.tiling}


    def hyperparams(self):
        return {'device':self.device, 'solver':self.solver, 'dt':self.dt}

        
    def _str(self, option_type):
        if option_type == 'all':
            return '_'.join([self._str(option) for option in ['W', 'dynamics', 'hyper']])

        else:
            if option_type == 'W':
                dic = self.W_options()
            if option_type == 'dynamics':
                dic = self.dynamics_options()
            if option_type == 'hyper':
                dic = self.hyperparams()

            return '_'.join([k + '=' + str(dic[k]) for k in dic.keys()])
        
        raise NotImplementedError(f"option type {option_type} not recognized")
