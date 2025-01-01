import torch
from skimage.filters import threshold_otsu
from scipy.ndimage import label, center_of_mass, gaussian_filter
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

def pca_torch_gpu(data, num_components):
    device = data.device
    centered_data = data - torch.mean(data, dim=0, keepdim=True)

    cov_matrix = torch.mm(centered_data.T, centered_data) / (data.size(0) - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov_matrix)

    sorted_indices = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    components = eigvecs[:, :num_components]
    explained_variance = eigvals[:num_components]

    projected_data = torch.mm(centered_data, components)

    return projected_data, explained_variance, components


def blob_center(traces, smoothing = 2, verb=False, sort_centers = False):
    t=1 if traces.ndim ==1 else len(traces)
    n = int(np.sqrt(traces.shape[-1]))
    
    vid = gaussian_filter(traces.reshape(-1, n, n), sigma=1, axes=(-1, -2))
    thresholds = np.array([threshold_otsu(img) for img in vid]) 
    binary_images = vid > thresholds[:, None, None] 


    labeled_images = np.zeros_like(binary_images, dtype=int)

    N= []
    if verb:
        print('labeling blobs')

    iterator = tqdm(range(len(binary_images))) if verb else range(len(binary_images))
    for i in iterator:
        labeled, num_features = label(binary_images[i])
        labeled_images[i] = labeled
        N.append(num_features)

    m = np.max(N)
    if verb:
        print(f'detected less than {m} blobs per frame')
        print(f'computing blob centers')


    iterator = tqdm(range(len(binary_images))) if verb else range(len(binary_images))
    centers = np.stack([np.array(center_of_mass(binary_images[i], labeled_images[i], range(1, m+1)))
           for i in iterator])  # TODO : remove divide warning
    
    centers = centers[:, :, [1, 0]] # need to transpose x, y for some reason
    dist = np.linalg.norm(centers-n/2, axis=2)
    
    if sort_centers : 
        sorted_indices = np.argsort(dist, axis=1)
        centers = np.take_along_axis(centers, sorted_indices[:, :, None], axis=1)

    pred = centers[np.arange(0, t, 1), np.nanargmin(dist, axis=1), :].squeeze()
    return pred, centers
    
    
def recover_speed_from_blob(center_coords, threshold = 3, verb=False):
    v_center = np.diff(center_coords, axis=0)
    v_norm = np.linalg.norm(v_center, axis=1)
    
    idx_jumps = np.where(v_norm>threshold)[0]
    if verb:
        print(f"found {len(idx_jumps)} best blob jumps : interpolating speed between them")

    v = v_center.copy()
    new_i = True
    
    for k in range(len(idx_jumps)):
        i = idx_jumps[k]
        
        if new_i :
            start = i-1
            
        if k<len(idx_jumps)-1 and i+1 == idx_jumps[k+1]:
            new_i = False
            
        else:   
            v[start+1: i+1] = (v[start] + v[i+1])/2
            new_i = True
        
    return v, idx_jumps

                     
def rescale_pos(true_pos, predicted_pos):
    model = LinearRegression(fit_intercept=False)
    true_pos = true_pos[:len(predicted_pos)]
    y = (true_pos - true_pos[0]).flatten() # start at origin
    X = predicted_pos.flatten().reshape(-1, 1)
    model.fit(X, y)
    prop_factor = model.coef_[0]
    rescaled_pos = prop_factor*predicted_pos + true_pos[0]
    r2 = model.score(X, y)
    return rescaled_pos, prop_factor, r2

    

def model_prediction(model_output, true_pos, silent=False, image_smoothing = 2, dt=1E-4, verb=True):
    best_center, _ = blob_center(model_output, smoothing = image_smoothing, verb=verb)
    predicted_speed, _ = recover_speed_from_blob(best_center, verb = verb)
    predicted_pos = np.cumsum(predicted_speed, axis=0)
    rescaled_pos, prop_factor, r2 = rescale_pos(true_pos, predicted_pos)
    if verb :
        print(rf"Estimated prediction to truth size ratio = {np.round(prop_factor, 4)}, R2={np.round(r2, 3)}")
    return rescaled_pos, prop_factor, r2

    
