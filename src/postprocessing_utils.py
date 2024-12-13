import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import lpips
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter
from skimage._shared.utils import _supported_float_type, check_shape_equality, warn
from skimage.util.arraycrop import crop
from skimage.util.dtype import dtype_range

HIGHER_BETTER = ['SSIM', 'luminance', 'contrast', 'structure']


class CustomMetrics():
    def __init__(self, metric_list, metric_kwargs=None):
        self.metric_list = metric_list
        if 'LPIPS' in self.metric_list:
            loss_fn = lpips.LPIPS(spatial=True)
            if torch.cuda.is_available():
                loss_fn = loss_fn.cuda()
            self.loss_fn = loss_fn

        if metric_kwargs is None:
            self.kwargs = {metric: {} for metric in self.metric_list}
        else:
            self.kwargs = metric_kwargs

        
    def MAE(self, input, recon, clip=True):
        if clip:
            difference =  input.clip(0,1) - recon.clip(0,1)
        else:
            difference = input - recon

        abs_diff = np.abs(difference)
        results = {
                   'values_2D': {'MAE': np.nanmean(abs_diff, axis=(1,2))},
                   'maps': {'MAE': abs_diff}, 
                   }
        
        return results


    def SSIM(self, input, recon, clip=True, data_range=1):
        if clip:
            input = input.clip(0,1)
            recon = recon.clip(0,1)

        if isinstance(input, torch.Tensor):
            input = input.cpu().numpy()
        if isinstance(recon, torch.Tensor):
            recon = recon.cpu().numpy()

        
        component_maps = {}
        
        for i in range(input.shape[0]):
            mssim_i, ssim_maps = structural_similarity_components(input[i],recon[i], data_range=data_range)
            for component in ssim_maps.keys():
                if component not in component_maps.keys():
                    component_maps[component] = []  
                component_maps[component].append(ssim_maps[component])
        
        results = {
            'values_3D': {},
            'values_2D' : {},
            'maps': {},
            }

    
        for component in component_maps.keys():
            this_comp_map = np.stack(component_maps[component], axis=0)

            results['maps'][component] = this_comp_map
            results['values_2D'][component] = np.nanmean(this_comp_map, axis=(1,2))
            
        return results
        

    def LPIPS(self, input, recon, batchsize=4, clip=True):
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)
            recon = torch.from_numpy(recon)
        if clip:
            input = input.clip(0,1)
            recon = recon.clip(0,1)

                
        input = (input*255).type(torch.uint8).unsqueeze(0)
        recon = (recon*255).type(torch.uint8).unsqueeze(0)
                
        input = input.permute(1,0,2,3)
        recon = recon.permute(1,0,2,3)

        
        input_img = input.repeat(1,3,1,1)
        recon_img = recon.repeat(1,3,1,1)
        
        if torch.cuda.is_available():
            input_img = input_img.cuda()
            recon_img = recon_img.cuda()



        distances = []
        for pos in range(0, input_img.shape[0], batchsize):
            start = pos 
            end = min(start + batchsize , input_img.shape[0])
            d =  self.loss_fn.forward(input_img[start:end], recon_img[start:end]).cpu().detach()

            distances.append(d)

        distances = torch.cat(distances)[:,0]
        distances_map = distances.numpy()

        results = {
            'values_2D': {'LPIPS': np.nanmean(distances_map, axis=(1,2))},
            'maps': {'LPIPS': distances_map},
        }
        
        return results



def structural_similarity_components(im1, im2,
                          *,
                          win_size=7,  
                          data_range=None, 
                          K1=0.01,
                          K2=0.03,
                          sigma=1.5):

    check_shape_equality(im1, im2)
    float_type = _supported_float_type(im1.dtype)


    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be positive")
    if np.any((np.asarray(im1.shape) - win_size) < 0):
        raise ValueError('win_size exceeds image extent.')
    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if data_range is None:
        if (np.issubdtype(im1.dtype, np.floating) or
            np.issubdtype(im2.dtype, np.floating)):
            raise ValueError(
                'Since image dtype is floating point, you must specify '
                'the data_range parameter. Please read the documentation '
                'carefully (including the note). It is recommended that '
                'you always specify the data_range anyway.')
        if im1.dtype != im2.dtype:
            warn("Inputs have mismatched dtypes. Setting data_range based on im1.dtype.",
                 stacklevel=2)
        dmin, dmax = dtype_range[im1.dtype.type]
        data_range = dmax - dmin
        if np.issubdtype(im1.dtype, np.integer) and (im1.dtype != np.uint8):
            warn("Setting data_range based on im1.dtype. " +
                 f"data_range = {data_range:.0f}. " +
                 "Please specify data_range explicitly to avoid mistakes.", stacklevel=2)

    ndim = im1.ndim

    filter_func = uniform_filter
    filter_args = {'size': win_size}

    # ndimage filters need floating point data
    im1 = im1.astype(float_type, copy=False)
    im2 = im2.astype(float_type, copy=False)

    NP = win_size ** ndim

    # filter has already normalized by NP
    cov_norm = NP / (NP - 1)  # sample covariance
    
    # compute (weighted) means
    ux = filter_func(im1, **filter_args)
    uy = filter_func(im2, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(im1 * im1, **filter_args)
    uyy = filter_func(im2 * im2, **filter_args)
    uxy = filter_func(im1 * im2, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)



    vx = vx.clip(0, None)
    vy = vy.clip(0, None)
    vxy = vxy.clip(0, None)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    C3 = C2 / 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                    2 * vxy + C2,
                    ux ** 2 + uy ** 2 + C1,
                    vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D

    luminance = (2 * ux * uy + C1) / (ux ** 2 + uy ** 2 + C1)
    structure = ( vxy + C3) / (vx + vy + C3)
    contrast = (2 * np.sqrt(vx) * np.sqrt(vy) + C2) / (vx + vy + C2)

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    # compute (weighted) mean of ssim. Use float64 for accuracy.
    mssim = crop(S, pad).mean(dtype=np.float64)

    
    maps = {'SSIM':S,
            'luminance':luminance, 
            'contrast': contrast, 
            'structure': structure,
    }


    return mssim, maps



def calculate_merics_and_snapshot2D(paths, my_metrics, input, recon, snapshots_folder=None):

    fnames = [Path(path).name for path in paths]
    # original_fnames = [fname.split('_')[0] + '.nii.gz' for fname in fnames]
    slices = [int(fname.split('_')[1].split('.')[0]) for fname in fnames]
    all_results_2D = [{'Filename': f, 'Slice': slice_i} for f, slice_i in zip(fnames, slices)]
    
    all_maps = {}
    for metric in my_metrics.metric_list:
        results = getattr(my_metrics, metric)(input, recon,  **my_metrics.kwargs[metric])  # keys: values, maps, labels

        all_maps.update(results['maps'])

        for pos_i in range(len(all_results_2D)):
            for metric_with_suffix in results['values_2D'].keys():
                all_results_2D[pos_i][metric_with_suffix] = results['values_2D'][metric_with_suffix][pos_i]
    
    if snapshots_folder is not None:
        plot_inputs = {'Image': input, 'Reconstruction': recon}
        os.makedirs(snapshots_folder, exist_ok=True)
        plot_overview2D(snapshots_folder, fnames, plot_inputs, all_maps)

    return  all_results_2D, all_maps

def plot_overview2D(savefolder, fnames, plot_inputs, all_maps):
    for sl, fname in enumerate(fnames):            
        all_keys = list(all_maps.keys())
        n_maps = len(all_keys)

        fig, ax = plt.subplots(1,len(plot_inputs)+n_maps, figsize=(len(plot_inputs)+n_maps*3, 3))
        for i, (label, img) in enumerate(plot_inputs.items()):
            obj = ax[i].imshow(img[sl,:,:], cmap='gray', vmin=0, vmax=1)
            plt.colorbar(obj, ax=ax[i],fraction=0.046)
            ax[i].axis('off')
            ax[i].set_title(label)


        for i, label in enumerate(all_keys):
            img = all_maps[label][sl,:,:]

            if label in HIGHER_BETTER:
                img = 1-img
            
            obj = ax[i+2].imshow(img, cmap='jet', vmin=0, vmax=1)
            plt.colorbar(obj, ax=ax[i+2],fraction=0.046)
            ax[i+2].axis('off')
            
            ax[i+2].set_title(label)
        plt.suptitle(f"{fname}")

        plt.tight_layout()
        plt.savefig(os.path.join(savefolder, f"{fname}.png"), bbox_inches='tight')
        #plt.show()
        plt.close()