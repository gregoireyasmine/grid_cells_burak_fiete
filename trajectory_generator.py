import numpy as np
from tqdm import tqdm
import os

TRAJ_DIR = os.getcwd() + '/sim_data/'
def avoid_wall(border_region, position, hd, box_width, box_height):
    '''
    Compute distance and angle to nearest wall
    '''
    x = position[:, 0]
    y = position[:, 1]
    dists = [box_width / 2 - x, box_height / 2 - y, box_width / 2 + x, box_height / 2 + y]
    d_wall = np.min(dists, axis=0)
    angles = np.arange(4) * np.pi / 2
    theta = angles[np.argmin(dists, axis=0)]
    hd = np.mod(hd, 2 * np.pi)
    a_wall = hd - theta
    a_wall = np.mod(a_wall + np.pi, 2 * np.pi) - np.pi

    is_near_wall = (d_wall < border_region) * (np.abs(a_wall) < np.pi / 2)
    turn_angle = np.zeros_like(hd)
    turn_angle[is_near_wall] = np.sign(a_wall[is_near_wall]) * (np.pi / 2 - np.abs(a_wall[is_near_wall]))

    return is_near_wall, turn_angle
    
def generate_trajectory(box_width, box_height, seq_len, batch_size, dt=1E-4, rot_vel_std=1.6, v_mean = 0.8,  border=0.03, border_slow = 0.25, load=False, save=False, silent=True):
    '''Generate a random walk in a rectangular box. Adapted from https://github.com/ganguli-lab/grid-pattern-formation'''
    
    fpath = TRAJ_DIR+f"trajectory_len={seq_len}_dt={dt}_w={box_width}_h={box_height}_rotvelstd={rot_vel_std}_meanv={v_mean}_border={border}_borderslow={border_slow}.npy"
    if load and os.path.exists(fpath):
        print(f"Found pre-computed trajectory at {fpath}, loading it")
        return np.load(fpath, allow_pickle = True).item()
    
    sigma = rot_vel_std * dt**0.5  # stdev rotation velocity (rads/sec)
    b = v_mean  # forward velocity rayleigh dist scale (m/sec) TODO: check scaling with dt
    mu = 0  # turn angle bias 
    border_region = border  # meters

    # Initialize variables
    position = np.zeros([batch_size, seq_len + 2, 2])
    head_dir = np.zeros([batch_size, seq_len + 2])
    position[:, 0, 0] = np.random.uniform(-box_width / 2, box_width / 2, batch_size)
    position[:, 0, 1] = np.random.uniform(-box_height / 2, box_height / 2, batch_size)
    head_dir[:, 0] = np.random.uniform(0, 2 * np.pi, batch_size)
    velocity = np.zeros([batch_size, seq_len + 2])

    # Generate sequence of random boosts and turns
    random_turn = np.random.normal(mu, sigma, [batch_size, seq_len + 1])
    random_vel = np.random.rayleigh(b, [batch_size, seq_len + 1])
    v = np.abs(np.random.normal(0, b * np.pi / 2, batch_size))

    iterator = range(seq_len + 1) if silent else tqdm(range(seq_len + 1))
    for t in iterator:
        # Update velocity
        v = random_vel[:, t]
        turn_angle = np.zeros(batch_size)

            # If in border region, turn and slow down
        is_near_wall, turn_angle = avoid_wall(border_region, position[:, t], head_dir[:, t], box_width, box_height)
        v[is_near_wall] *= border_slow

        # Update turn angle
        turn_angle += random_turn[:, t]

        # Take a step
        velocity[:, t] = v * dt

        update = velocity[:, t, None] * np.stack([np.cos(head_dir[:, t]), np.sin(head_dir[:, t])], axis=-1)
        position[:, t + 1] = position[:, t] + update

        # Rotate head direction
        head_dir[:, t + 1] = head_dir[:, t] + turn_angle

    head_dir = np.mod(head_dir + np.pi, 2 * np.pi) - np.pi 

    traj = {}
    
    # Input variables
    traj['init_hd'] = head_dir[:, 0, None]
    traj['init_x'] = position[:, 1, 0, None]
    traj['init_y'] = position[:, 1, 1, None]

    traj['ego_v'] = velocity[:, 1:-1]
    ang_v = np.diff(head_dir, axis=-1)
    traj['phi_x'], traj['phi_y'] = np.cos(ang_v)[:, :-1], np.sin(ang_v)[:, :-1]

    # Target variables
    traj['target_hd'] = head_dir[:, 1:-1]
    traj['target_x'] = position[:, 2:, 0]
    traj['target_y'] = position[:, 2:, 1]

    if save:
        np.save(fpath, traj)

    return traj
