import matplotlib.pyplot as plt
import torch
import numpy as np
# import cv2
import tqdm 
from analysis import *
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
import os
import cv2
from matplotlib.colors import ListedColormap


FIG_PATH = os.getcwd() + '/figures/'

plt.rcParams.update({
    'xtick.labelsize': 8,  # Tick label font size (x-axis)
    'ytick.labelsize': 8,  # Tick label font size (y-axis)
    'legend.fontsize': 8, # Legend font size
    'axes.titlesize': 10,  # Optional: Adjust title size
    'axes.labelsize': 8   # Optional: Adjust axes label size
})
mpl.rcParams['figure.dpi'] = 150


def plot_10_frames(traj, dt, plot_center = True, plot_blob_centers=True):
    n = int(np.sqrt(len(traj[0])))
    fig, ax = plt.subplots(2, 5, figsize = (20, 10))
    times = np.linspace(0, len(traj) -1, 10, dtype = 'int')
    for k, t in enumerate(times):
        ax[k//5, k%5].imshow(traj[t, :].reshape(n, n))
        ax[k//5, k%5].set_title(f"{t*1000*dt} ms")
        if plot_center:
            ax[k//5, k%5].axvline(n/2, c='red', lw=0.8)
            ax[k//5, k%5].axhline(n/2, c='red', lw=0.8)
        if plot_blob_centers:
            center_coords, p = blob_center(traj[t, :], smoothing = 1, verb=False, sort_centers = True)
            ax[k//5, k%5].scatter(p[0, 1:, 0], p[0, 1:, 1], c='red', marker='x')
            ax[k//5, k%5].scatter(p[0, 0, 0], p[0, 0, 1], c='blue', marker='+')
    plt.show()


def rec_movie(traces, fpath, dt, speed_ratio=1/4, plot_center=True, plot_blob_centers=False, frame_labels=None, cmap='magma'):
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    from tqdm import tqdm

    # Paramètres de la vidéo
    fps = int(1 / dt) * speed_ratio  # frames per second
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec

    n = int(np.sqrt(traces.shape[1]))
    fig, ax = plt.subplots()
    ax.axis('off')  # Supprime la bordure de l'axe
    ax.set_xticks([])  # Supprime les ticks sur l'axe x
    ax.set_yticks([])  # Supprime les ticks sur l'axe y
    plt.tight_layout()

    generated_images = []

    print('Generating video...')

    height, width = None, None
    
    for t, frame in tqdm(enumerate(traces.reshape(-1, n, n))):
        ax.clear()  # Réinitialiser l'axe au lieu de le recréer
        ax.imshow(frame, cmap=cmap, interpolation=None)
        ax.set_xticks([])  # Supprime les ticks sur l'axe x
        ax.set_yticks([])  # Supprime les ticks sur l'axe y

        if plot_center:
            ax.axvline(n / 2, c='red', lw=0.8)
            ax.axhline(n / 2, c='red', lw=0.8)
            
        if plot_blob_centers:
            center_coords, p = blob_center(traces[t, :], smoothing=1, verb=False, sort_centers=True)
            ax.scatter(p[0, 1:, 0], p[0, 1:, 1], c='red', marker='x')
            ax.scatter(p[0, 0, 0], p[0, 0, 1], c='blue', marker='+') 

        if frame_labels is not None:
            ax.set_title(frame_labels[t], fontsize=12)  # Ajoute le titre avec une taille de police réduite
        
        plt.subplots_adjust(top=0.9)  # Réduit la marge supérieure pour donner plus d'espace pour le titre
        
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Format (hauteur, largeur, 4)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        if height is None or width is None:
            height, width, _ = image.shape
            video_writer = cv2.VideoWriter(fpath, fourcc, fps, (width, height))

        video_writer.write(image)  # Écriture directe dans la vidéo

    video_writer.release()  # Finaliser l'écriture de la vidéo
    plt.close(fig)
    print(f"Video recorded at {fpath}")
    return None



def plot_4d_projection(traces, basis=None, eigval=None, dpi=100):
    if basis is None:
        projections, exp_var, basis = pca_torch_gpu(traces, 4)
    else:
        projections = traces @ basis[:, :4]
        exp_var = eigval[:4] / eigval.sum() if eigval is not None else np.ones(4) / 4

    trajectory = projections.cpu().numpy()

    fig = plt.figure(figsize=(10, 4), dpi=dpi)

    ax = fig.add_subplot(121, projection='3d')
    p = ax.scatter(xs=trajectory[:, 0].astype('float'),
                   ys=trajectory[:, 1].astype('float'),
                   zs=trajectory[:, 2].astype('float'),
                   c=trajectory[:, 3].astype('float'),
                   cmap=plt.cm.magma, s=1)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    cbar = fig.colorbar(p, ax=ax, pad=0.15, label='PC4')
    cbar.set_ticks([])

    # Create bar plot for explained variance
    ax2 = fig.add_subplot(122)
    components = ['PC1', 'PC2', 'PC3', 'PC4']
    ax2.bar(components, exp_var, color='skyblue', alpha=0.7)
    ax2.set_ylabel('Explained Variance')
    ax2.set_ylim(0, 1)
    ax2.set_title('Explained Variance Ratio')

    plt.tight_layout()
    plt.show()


def plot_cell_connectivity(grid, ax, i=None, title='Example cell connectivity', cbar=True, cmap='viridis'):
    
    if i is None: 
        i = torch.randint(0, grid.n**2, (1, ))
        
    im = ax.imshow(grid.W.cpu()[i].reshape(grid.n, grid.n), cmap=cmap)
    
    ax.scatter(i % grid.n, i // grid.n, marker='x', c='red', label='Focal cell location')
    
    ax.legend(loc='upper right', frameon=True, fontsize=8)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_title(title)
    
    if cbar:
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
        cbar.set_label('Weight', rotation=270, labelpad=10)
        
    return None

def plot_s0(grid, ax, title='Resting state activity', cbar=True, cmap='magma'):
    
    im = ax.imshow(grid.s0.cpu().reshape(grid.n, grid.n), cmap=cmap)
        
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_title(title, fontsize=10)

    if cbar:
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
        cbar.set_label('Cell activation', rotation=270, labelpad=10)
        
    return None
    


def network_presentation(grid, show=True, save=True):
    
    fig = plt.figure(figsize=(12, 3))
    
    spec = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 1.15], wspace=0.01)  # Small space between plots 2, 3, and 4
    
    ax0 = fig.add_subplot(spec[0, 0])
    plot_s0(grid, ax0, cbar=True)

    for i in range(3):
        ax = fig.add_subplot(spec[0, i + 1])
        plot_cell_connectivity(grid, ax, cbar=(i == 2))

    plt.subplots_adjust(left=0.0, right=0.95, bottom=0.1, top=0.9, wspace=0.4)
    
    if show:
        plt.show()

    if save :
        plt.savefig(FIG_PATH+"network_presentation.png", transparent=True)
    return fig, [fig.axes[i] for i in range(4)]


def plot_trajectory(loc_x, loc_y, box_width=None, box_height=None, ax=None, show=False, show_limits = True, plot_kwargs={}):
    
    if ax is None:
        fig, ax = plt.subplots()
        
    ax.plot(loc_x, loc_y, **plot_kwargs)
    
    if box_width is not None:
        ax.set_xlim(-box_width/2, box_width/2)
        
    if box_height is not None:
        ax.set_ylim(-box_height/2, box_height/2)

    if show_limits : 
        ax.scatter(loc_x[0], loc_y[0], c='blue', s=10, marker='o', label='Initial position')
        ax.scatter(loc_x[-1], loc_y[-1], c='green', s=10, marker='o', label='Final position')

    ax.set_xticks([])
    ax.set_yticks([])  

    if show :
        plt.show()

    return ax


def compare_model_prediction(predicted_position, true_position, ax = None, box_width=None, show_limits = True, box_height=None, show=True):

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4), dpi = 200)
    
    kw = {'c': 'black', 'ls': '-', 'lw':1.4, 'label':'True position'}
    ax = plot_trajectory(*true_position.T, box_width = box_width, box_height = box_height,ax=ax, show=False, show_limits = show_limits, plot_kwargs=kw)
    
    kw = {'c': 'red', 'ls': ':', 'lw':0.7, 'label':'Model prediction'}
    ax = plot_trajectory(*predicted_position.T, box_width = box_width, box_height = box_height,ax=ax, show=False, show_limits = False, plot_kwargs=kw)
    
    plt.legend()
    if show :
        plt.show()

    return ax


def show_place_cell_activity(place_cells, box_width=None, idx = None, ax=None, show=True, cbar = True, cmap = 'magma'):
    
    if idx is None :
        idx = np.random.randint(0, place_cells.sheet_size**2)

    loc = place_cells.loc_box.unsqueeze(1).to(place_cells.device)

    activity = place_cells.out(loc)[idx].cpu().reshape(place_cells.sheet_size, place_cells.sheet_size)

    if ax is None : 
        fig, ax = plt.subplots()

    im= ax.imshow(activity, cmap = cmap)
    
    if cbar:
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', ticks=None, pad=0.02, shrink=0.8)
        cbar.set_label('Cell activation', rotation=270, labelpad=10)

    
    ax.set_title('Arbitrary place cell activation')

    if box_width is not None : 
        ax.set_xticks([0, (place_cells.sheet_size-1)/2, place_cells.sheet_size-1])
        ax.set_yticks([0, (place_cells.sheet_size-1)/2, place_cells.sheet_size-1])   
 
        ax.set_xticklabels([-box_width/2, 0, box_width/2])
        ax.set_yticklabels([-box_width/2, 0, box_width/2])

        ax.set_xlabel('Animal x location (m)')
        ax.set_ylabel('Animal y location (m)')
    
    if show:
        plt.show()
        
    return ax


def present_training_environment(network, trainer, num_trajectories = 10, show=True):
    
    fig, ax = plt.subplots(1, 2, figsize = (10, 4))
    
    show_place_cell_activity(network.place_cells, box_width=network.options.box_width, ax=ax[0], show=False)
    
    trajectories = next(trainer.generator.generator())[1].cpu()
    
    for k in range(num_trajectories):
        kw = {'c': 'black', 'ls': '-', 'lw':1.4}
        plot_trajectory(*trajectories[:, k,:].T, ax=ax[1], box_width = network.options.box_width, box_height = network.options.box_width, plot_kwargs = kw)

    handles, labels = ax[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax[1].legend(unique.values(), unique.keys())
    
    ax[1].set_aspect('equal')
    ax[1].set_title('Example training trajectories')

    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig, ax


def show_training_results(trainer, model, num_trajectories = 10000, sequence_length = 20, plotted_trajectories = 10, dt=1e-2, show=True, save=False):
    
    fig, ax = plt.subplots(1, 2, figsize = (7, 3.5), dpi=200)
        
    inputs, trajectories = trainer.generator.batch(n_trajectories = num_trajectories, sequence_length = sequence_length, dt=dt)
    
    predictions, err = model.test_pred(inputs, trajectories)
    trajectories = trajectories.cpu()
    sem = err.std(1)/(num_trajectories**0.5)
    err = err.mean(1)

    for k in range(plotted_trajectories):
        ax[0] = compare_model_prediction(predictions[:, k, :], trajectories[:, k, :], ax = ax[0], box_width = model.options.box_width,
                                         show_limits = True, show=False)
        
    handles, labels = ax[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax[0].legend(unique.values(), unique.keys(), loc = 'upper right')

    ax[1].plot(err*100)
    ax[1].fill_between(range(sequence_length), (err-sem)*100, (err+sem)*100, alpha = 0.2)
    ax[1].set_xlabel('time step in trajectory')
    ax[1].set_ylabel('average error (cm)')
    # ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.tight_layout()
    if save:
        plt.savefig(FIG_PATH+f"training_results_seq_len_{sequence_length}_dt_{dt}.png", transparent=True)

    
    if show:
        plt.show()

    
    return fig, ax


def create_alpha_colormap(alpha_low=0.0, alpha_high=0.8, color = np.array([[0.8, 0., 0.1, 1]])):
    alpha_values = np.linspace(alpha_low, alpha_high, 256)
    
    colors = np.tile(color, (256, 1))
    colors[:, -1] = alpha_values  
    
    return ListedColormap(colors)



def plot_rate_over_trajectory(pos, activity, idx=None, show=True, save=True):
    fig, ax = plt.subplots(figsize = (4, 4))
    
    custom_cmap = create_alpha_colormap(alpha_low=0.0, alpha_high=1.0)
    ax.set_facecolor('black')

    if idx is None:
        idx = np.random.randint(activity.shape[1])

    plt.scatter(*pos.T, c=activity[:, 23], cmap=custom_cmap, label = 'firing locations')

    
    ax.plot(*pos.T, lw=0.5, label = 'mice location')
    plt.legend(loc='upper right')

    if save:
        plt.savefig(FIG_PATH+'rate_trajectory.png', transparent=True)

    if show:
        plt.show()
    
    return fig, ax

    
