import torch
import os
from trajectory_generator import generate_trajectory
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

class Place_cells():
    """
    Regular deterministic sheet_size*sheet_size tiling of place cells on the square arena.
    Cells activate proportionally to the distance between their preferred location and the animal position.
    """
    
    def __init__(self, sheet_size, sigma, space_size, device):
        '''
        Args:
            sheet_size: Number of cells on the grid
            sigma : size of the gaussian activation of place cells around the centered position
            space_size : width of the square arena
        '''
        self.sheet_size = sheet_size
        self.cell_idx = torch.arange(0, sheet_size**2, 1)
        self.device = device
        
        self.loc_sheet = torch.stack([self.cell_idx%sheet_size, self.cell_idx//sheet_size], axis = -1).to(device)
        self.loc_box = (self.loc_sheet/self.sheet_size - 1/2) * space_size
        
        self.sigma = sigma
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)

    def out(self, pos):
        """ Return grid cell pattern activation """
        dist = ((pos[:, :, None, :] - self.loc_box[None, None, :])**2).sum(axis=-1)
        return self.softmax(- self.sigma**(-2) * dist)
        
    def logout(self, pos):
        """ Return log grid cell pattern activation (for computational stability during training)"""
        dist = ((pos[:, :, None, :] - self.loc_box[None, None, :])**2).sum(axis=-1)
        return self.logsoftmax(- self.sigma**(-2) * dist)


class RNN_options():
    """
    Options to pass to the TrainableNetwork class.
    """

    def __init__(self,):
        self.save_dir = os.getcwd() + '/models/'
        self.n_steps = 100000         # number of training steps
        self.batch_size = 200         # number of trajectories per batch
        self.sequence_length = 20     # number of steps in trajectory
        self.learning_rate = 1e-4     # gradient descent learning rate
        self.pc_sheet_size = 23       # sqrt of number of place cells 
        self.gc_sheet_size = 40       # sqrt of number of grid cells
        self.pc_sigma = 0.12          # width of place cell center tuning curve (m)
        self.activation = 'tanh'      # recurrent nonlinearity
        self.weight_decay = 1e-4      # strength of weight decay on recurrent weights
        self.box_width = 2.2          # width of square arena 
        self.dt = 1E-2
        self.optimizer = 'adam'       # adam or rmsprop
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.clip_grad = False
        self.debug = False

    def _str(self,):
       attributes = vars(self).copy()
       del attributes['save_dir']
       del attributes['debug']
       return "_".join([f"{key}_{value}" for key, value in attributes.items()])

    
class TrainableNetwork(torch.nn.Module):
    """
    Simple architecture (gridcell input -> linear encoder -> RNN -> linear decoder -> gridcell output]
    Encoder and decoder allow to switch between grid cells representations of space to place cells representations
    RNN should learn an abstract representation of space that allows to integrate velocity inputs.
    """
    def __init__(self, options):
        super().__init__()
        self.options = options 
        
        self.n_gc = options.gc_sheet_size**2
        self.n_pc = options.pc_sheet_size**2
        
        self.place_cells = Place_cells(options.pc_sheet_size, options.pc_sigma, options.box_width, options.device)

        self.encoder = torch.nn.Linear(self.n_pc, self.n_gc, bias=False).to(options.device)
        self.RNN = torch.nn.RNN(input_size=2,
                                hidden_size=self.n_gc,
                                nonlinearity=options.activation,
                                bias=False).to(options.device)
        # Linear read-out weights
        self.decoder = torch.nn.Linear(self.n_gc, self.n_pc, bias=False).to(options.device)
        
        self.softmax = torch.nn.Softmax(dim=-1)
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    
    def forward(self, inputs):
        v, p0 = inputs

        pc_0 = self.place_cells.out(p0)
        init_state = self.encoder(pc_0)
        g, hidden = self.RNN(v, init_state)
        scores = self.decoder(g)
        if self.options.debug:
            print(f"pc0 ranging in : {pc_0.min(), pc_0.max()}")
            print(f"init_state ranging in : {(init_state.min(), init_state.max())}")
            print(f"v ranging in : {(v.min(), v.max())}")
            print(f"g ranging in : {(g.min(), g.max())}")
            
        logpred = self.logsoftmax(scores)
        return logpred, scores, hidden, g, init_state
        
    def loss(self, logpred, pos):
        true_gc = self.place_cells.out(pos)

        loss = -(true_gc*logpred).sum(-1).mean() # CE Loss with precomputed log (numerical stability)
        # Weight regularization 
        loss += self.options.weight_decay * (self.RNN.weight_hh_l0**2).sum()

        # Compute decoding error
        pred_pos = self.place_cells.loc_box[torch.argmax(logpred, axis=-1)] 
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

        return loss, err

    def test_pred(self, inputs, pos):
        with torch.no_grad():
            
            logpred = self.forward(inputs)[0]
            
            pred_pos = self.place_cells.loc_box[torch.argmax(logpred, axis=-1)]
            
            err = torch.sqrt(((pos - pred_pos)**2).sum(-1))
            
        return pred_pos.cpu(), err.cpu()
        

class TrajectoryGenerator():
    """
    Generates batched trajectories using the generate_trajectory function in the trajectory_generator.py file.
    Each trajectory encountered by the network is new. 
    NB : Any non-uniformity in the sampled animal locations distribution is likely to bias the network and make stick it in a local loss minima.
    """
    
    def __init__(self, model):
        self.box_width = model.options.box_width
        self.box_height = model.options.box_width
        self.sequence_length = model.options.sequence_length
        self.batch_size = model.options.batch_size
        self.dt = model.options.dt
        self.device = model.options.device


    def batch(self, n_trajectories, sequence_length, dt):
        traj = generate_trajectory(self.box_width, self.box_height, sequence_length, n_trajectories, dt)

        v = traj['input_v']
        v = torch.tensor(v, dtype=torch.float32).transpose(0, 1).to(self.device)

        pos = np.stack([traj['target_x'], traj['target_y']], axis=-1)
        pos = torch.tensor(pos, dtype=torch.float32).transpose(0, 1).to(self.device)
            
        init_pos = np.concatenate([traj['init_x'], traj['init_y']], axis=1)
        init_pos = torch.tensor(init_pos, dtype=torch.float32).to(self.device)

        inputs = (v, init_pos[None])
        return inputs, pos
        
    def generator(self):
        while True:
            yield self.batch(self.batch_size, self.sequence_length, self.dt)
        

class Trainer(object):
    
    def __init__(self, model, restore=True):
        self.options = model.options
        self.model = model
        lr = self.options.learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) if self.options.optim == 'adam' else torch.optim.RMSprop(self.model.parameters(), lr=lr)
        

        self.generator = TrajectoryGenerator(model)
        self.loss = []
        self.err = []

        # Set up checkpoints
        self.ckpt_dir = os.path.join(self.options.save_dir, self.options._str())
        ckpt_path = os.path.join(self.ckpt_dir, 'most_recent_model.pth')
        if restore and os.path.isdir(self.ckpt_dir) and os.path.isfile(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, weights_only = True))
            print("Restored trained model from {}".format(ckpt_path))
        else:
            if not os.path.isdir(self.ckpt_dir):
                os.makedirs(self.ckpt_dir, exist_ok=True)
            print("Initializing new model from scratch.")
            print("Saving to: {}".format(self.ckpt_dir))

    
    def train_step(self, inputs, pos):
        self.model.zero_grad()
        logpred = self.model(inputs)[0]
        
        if self.options.debug:
            print(f"logpred ranging in ({logpred.min()}, {logpred.max()})")
            print(f"pos ranging in ({pos.min()}, {pos.max()})")
            
        loss, err = self.model.loss(logpred, pos)
        if self.options.debug:
            print("loss :", loss)

        loss.backward()
        self.optimizer.step()
        return loss, err.item()


    def plot_loss_err(self):
        clear_output(wait=True)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(np.log(self.loss))
        ax[0].set_title("Log Loss")
        ax[1].plot(self.err*100)
        ax[1].set_title("Error (cm)")
        ax[0].set_xlabel('Step')
        ax[1].set_xlabel('Step')
        plt.show()

    
    def train(self, n_epochs: int = 1000, n_steps=10000, save=True, plot=True, plot_every = 50):

        gen = self.generator.generator()
        
        for epoch_idx in range(n_epochs):
            
            tbar = tqdm(range(n_steps), leave=False)
            for step_idx in tbar:
                
                inputs, pos = next(gen) 
                
                loss, err = self.train_step(inputs, pos)

                if self.options.debug :
                     print(f"Total grad norm = {self._gradient_norm_()} ")

                if torch.isnan(loss):
                    break

                else :
                    loss = loss.item()
                    self.loss.append(loss)
                    self.err.append(err)

                tbar.set_description(f'Epoch {epoch_idx}/{n_epochs} step {step_idx} : Loss {loss:.2f} Error = {100*err:5.1f} cm')

                if plot and (step_idx+1)%plot_every == 0 :
                    self.plot_loss_err()

            if save:
                # Save checkpoint
                ckpt_path = os.path.join(self.ckpt_dir, 'epoch_{}.pth'.format(epoch_idx))
                torch.save(self.model.state_dict(), ckpt_path)
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, 'most_recent_model.pth'))

    def _gradient_norm_(self):
        "a debug function to compute gradient norm"
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm