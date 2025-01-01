import torch
import os
from trajectory_generator import generate_trajectory
from tqdm import tqdm
import numpy as np

class Place_cells():
    # uniform deterministic n*n tiling of place cells on the square arena
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
        self.loc = torch.stack([self.cell_idx%sheet_size, self.cell_idx//sheet_size], axis = -1).to(device)
        self.sigma = sigma * sheet_size * space_size
        self.softmax = torch.nn.Softmax(dim=-1)

    def out(self, pos):
        """ Return grid cell pattern activation """
        dist = ((pos[:, :, None, :] - self.loc[None, None, :])**2).sum(axis=-1)
        return self.softmax(- self.sigma**(-2) * dist)


class RNN_options():
    def __init__(self,):
        self.save_dir = os.getcwd() + '/models/'
        self.n_steps = 100000         # number of training steps
        self.batch_size = 200         # number of trajectories per batch
        self.sequence_length = 20     # number of steps in trajectory
        self.learning_rate = 1e-5     # gradient descent learning rate
        self.pc_sheet_size = 23      # sqrt of number of place cells 
        self.gc_sheet_size = 40       # sqrt of number of grid cells
        self.pc_sigma = 0.12          # width of place cell center tuning curve (m)
        self.activation = 'tanh'      # recurrent nonlinearity
        self.weight_decay = 1e-4      # strength of weight decay on recurrent weights
        self.box_width = 2.2            # width of square arena 
        self.dt = 1E-2
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'

    def _str(self,):
       attributes = vars(self).copy()
       del attributes['save_dir']
       return "_".join([f"{key}_{value}" for key, value in attributes.items()])
    
class TrainableNetwork(torch.nn.Module):
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

    
    def forward(self, inputs):
        v, p0 = inputs
        init_state = self.encoder(self.place_cells.out(p0))
        g, hidden = self.RNN(v, init_state)
        scores = self.decoder(g)
        pred = self.softmax(scores)
        return pred, scores, hidden, g, init_state


    def loss(self, pred, pos):
        true_gc = self.place_cells.out(pos)

        loss = -(true_gc*torch.log(pred)).sum(-1).mean()
        # Weight regularization 
        loss += self.options.weight_decay * (self.RNN.weight_hh_l0**2).sum()

        # Compute decoding error
        pred_pos = self.place_cells.loc[torch.argmax(pred, axis=-1)]
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

        return loss, err


class TrajectoryGenerator():
    def __init__(self, model):
        self.box_width = model.options.box_width
        self.box_height = model.options.box_width
        self.sequence_length = model.options.sequence_length
        self.batch_size = model.options.batch_size
        self.dt = model.options.dt
        self.device = model.options.device

    def generator(self):
        while True:
            traj = generate_trajectory(self.box_width, self.box_height, self.sequence_length, self.batch_size, self.dt)

            v = np.stack([traj['ego_v'] * np.cos(traj['target_hd']), traj['ego_v'] * np.sin(traj['target_hd'])], axis=-1)
            v = torch.tensor(v, dtype=torch.float32).transpose(0, 1).to(self.device)

            pos = np.stack([traj['target_x'], traj['target_y']], axis=-1)
            pos = torch.tensor(pos, dtype=torch.float32).transpose(0, 1).to(self.device)
            
            init_pos = np.concatenate([traj['init_x'], traj['init_y']], axis=1)
            init_pos = torch.tensor(init_pos, dtype=torch.float32).to(self.device)

            inputs = (v, init_pos[None])

            yield inputs, pos
        
    

class Trainer(object):
    def __init__(self, model, restore=True):
        self.options = model.options
        self.model = model
        lr = self.options.learning_rate
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr)

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
        pred = self.model(inputs)[0]
        loss, err = self.model.loss(pred, pos)
        loss.backward()
        self.optimizer.step()
        return loss.item(), err.item()

    
    def train(self, n_epochs: int = 1000, n_steps=10000, save=True):

        tbar = tqdm(range(n_steps), leave=False)

        gen = self.generator.generator()
        
        for epoch_idx in range(n_epochs):
            for step_idx in range(n_steps):
                
                inputs, pos = next(gen) 
                
                loss, err = self.train_step(inputs, pos)
                self.loss.append(loss)
                self.err.append(err)

                tbar.set_description(f'Epoch {epoch_idx}/{n_epochs} step {step_idx}/{n_steps} : Loss {loss:.2f} Error = {100*err:5.1f} cm')

            if save:
                # Save checkpoint
                ckpt_path = os.path.join(self.ckpt_dir, 'epoch_{}.pth'.format(epoch_idx))
                torch.save(self.model.state_dict(), ckpt_path)
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, 'most_recent_model.pth'))
        