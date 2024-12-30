import torch
import numpy as np


def deterministic_theta_tiling(n):
    thetas = [0, np.pi/2, np.pi, -np.pi/2] # W N E S
    theta_pref = torch.tensor([thetas*(n//4), thetas[2:] + thetas*(n//4 - 1) + thetas[:2]]*(n//2))
    return theta_pref


def random_theta_tiling(n):
    thetas = torch.tensor([0, np.pi/2, np.pi, -np.pi/2]) # W N E S
    dir_pref = np.random.randint(0, 4, n*n)
    theta_pref = thetas[[dir_pref]]
    return theta_pref.reshape(n, n)