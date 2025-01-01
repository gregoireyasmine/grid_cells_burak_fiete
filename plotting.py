import matplotlib.pyplot as plt
import torch
import numpy as np
# import cv2
import tqdm 
from analysis import *
import matplotlib.gridspec as gridspec


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
            center_coords, p = blob_center(traj[t, :], n_candidates=5, smoothing = 1, verb=False, sort_centers = True)
            ax[k//5, k%5].scatter(p[0, 1:, 0], p[0, 1:, 1], c='red', marker='x')
            ax[k//5, k%5].scatter(p[0, 0, 0], p[0, 0, 1], c='blue', marker='+')
    plt.show()


def rec_movie(traces, fpath, dt, speed_ratio=1/4, cmap='magma'):
# Paramètres de la vidéo
    fps = int(1/dt) * speed_ratio  # frames per second
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec

    n = int(np.sqrt(traces.shape[1]))
    fig, ax = plt.subplots()
    ax.axis('off')  
    plt.tight_layout()

    generated_images = []

    print('generating movie...')
    for frame in tqdm(traces.reshape(-1, n, n)):
        ax.imshow(frame, cmap=cmap, interpolation=None) 
        fig.canvas.draw() 

        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Format (hauteur, largeur, 4)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR) 
        
        generated_images.append(image)

        plt.close(fig)  

    height, width, _ = generated_images[0].shape
    video_writer = cv2.VideoWriter(fpath, fourcc, fps, (width, height))

    for img in generated_images:
        video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

    video_writer.release()
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
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)

    if cbar:
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
        cbar.set_label('Cell activation', rotation=270, labelpad=10)
    return None
    


def network_presentation(grid, show=True):
    # Create a figure and define the grid specification with adjusted width ratios
    fig = plt.figure(figsize=(12, 3))
    
    # Define GridSpec with unequal column widths
    spec = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 1.15], wspace=0.01)  # Small space between plots 2, 3, and 4
    
    # Plot the first figure (plot_s0)
    ax0 = fig.add_subplot(spec[0, 0])
    plot_s0(grid, ax0, cbar=True)

    # Plot the similar plots (2, 3, and 4)
    for i in range(3):
        ax = fig.add_subplot(spec[0, i + 1])
        plot_cell_connectivity(grid, ax, cbar=(i == 2))

    # Adjust margins: Add extra space between plot 1 and plot 2, but keep others close
    plt.subplots_adjust(left=0.0, right=0.95, bottom=0.1, top=0.9, wspace=0.4)
    
    if show:
        plt.show()
    return fig, [fig.axes[i] for i in range(4)]


def plot_trajectory(loc_x, loc_y, box_width=None, box_height=None, ax=None, show=False, plot_kwargs={}):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(loc_x, loc_y, **plot_kwargs)
    if box_width is not None:
        ax.set_xlim(-box_width/2, box_width/2)
    if box_height is not None:
        ax.set_ylim(-box_height/2, box_height/2)
    if show :
        plt.show()
    return ax


def compare_model_prediction(predicted_position, true_position, box_width=None, box_height=None, show=True):
    fig, ax = plt.subplots(figsize=(5, 5), dpi = 200)
    kw = {'c': 'black', 'ls': '-', 'lw':1.4, 'label':'True position'}
    ax = plot_trajectory(*true_position.T, box_width = box_width, box_height = box_height,ax=ax, show=False, plot_kwargs=kw)
    kw = {'c': 'red', 'ls': ':', 'lw':0.7, 'label':'Model prediction'}
    ax = plot_trajectory(*predicted_position.T, box_width = box_width, box_height = box_height,ax=ax, show=False, plot_kwargs=kw)
    ax.scatter(*true_position[0], c='blue', s=10, marker='o', label='Initial position')
    ax.scatter(*true_position[-1], c='green', s=10, marker='o', label='Final position')
    plt.legend()
    plt.show()
