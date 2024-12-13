import torchio as tio
import copy
import numpy as np
import random
import raster_geometry as rg

from torchio import RandomMotion, RandomSpike, RandomSwap, RandomElasticDeformation, RandomBlur, RandomBiasField, RandomGhosting, RandomNoise
from typing import Tuple

import numpy as np
import torch

def define_kwargs(types, n_files):
    kwarg_options = {}
    for augm_type, options in types:
        kwarg_options[augm_type] = options
        for key, values in options.items():
            if isinstance(values, np.ndarray):
                values = list(values)
            elif isinstance(values, int) or isinstance(values, float):
                values = [values]
            
            n_copies = n_files // len(values)
            assert n_files % len(values) == 0, f"Number of files ({n_files}) should be a multiple of the number of values ({len(values)})"

            all_values = values * n_copies
            random.shuffle(all_values)
            
            kwarg_options[augm_type][key] = all_values
    return kwarg_options
    



def set_seed(random_seed):
    # Control randomness for reproduction
    if random_seed != None or random_seed != -1:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)

class CustomTransform(tio.Transform):
    def __init__(self):
        super().__init__()

    def apply_transform(self, subject):
        img = subject['image'].data
        if img.shape[0] != 1:
            raise ValueError('This transform only works with a batchsize of 1')
        
        img = img[0]
        new_img = self.apply_this_transform(img).unsqueeze(0)

        transformed_subject = copy.deepcopy(subject)
        transformed_subject['image'].data = new_img

        return transformed_subject
    

    

class ToyLesion(CustomTransform):
    """ To create a 3D toy lesion in the 3D volume"""
    def __init__(self, radius, intensity, smoothing=True):
        super().__init__()
        self.radius = radius
        self.intensity = intensity
        self.smoothing = smoothing

    def apply_this_transform(self, img):

        border = np.zeros(img.shape)
        border[self.radius:-self.radius, 
               self.radius:-self.radius, 
               self.radius:-self.radius] = 1
        body_mask = np.where(img==0, 0 , 1) * border
        position = body_mask.nonzero()
        coord_pos = np.random.randint(0, len(position[0]))
        mid_coord = [position[dim][coord_pos] for dim in range(3)]
        position = np.array(mid_coord)/np.array(img.shape)

        sphere = rg.sphere(img.shape,
                           self.radius, 
                           position= position, 
                           smoothing= self.smoothing)

        return img *(1-sphere) + sphere * self.intensity


class Circle2D(CustomTransform):
    """ To create a 2D circle in the 2D slice"""
    def __init__(self, radius, intensity, smoothing=True,):
        super().__init__()
        self.radius = radius
        self.intensity = intensity
        self.smoothing = smoothing

    def apply_this_transform(self, img):
        img = img.squeeze()
        border = np.zeros(img.shape)
        border[self.radius:-self.radius, 
               self.radius:-self.radius] = 1
        body_mask = np.where(img==0, 0 , 1) * border
        position = body_mask.nonzero()
        coord_pos = np.random.randint(0, len(position[0]))
        mid_coord = [position[dim][coord_pos] for dim in range(2)]
        position = np.array(mid_coord)/np.array(img.shape)

        circle = rg.circle(img.shape,
                           self.radius, 
                           position= position, 
                           smoothing= self.smoothing)


        output =  img *(1-circle) + circle * self.intensity
        output = output.unsqueeze(2)
        return output


class BlackSlice(CustomTransform):
    def __init__(self, n_slices, ax):
        super().__init__()
        self.n_slices = n_slices
        self.ax = ax

    def apply_this_transform(self, img):
        new_img = copy.deepcopy(img)
        NZ = img.nonzero()[:,self.ax]
        start = random.choice(range(NZ.min(), NZ.max() - self.n_slices + 1)) 
        end = start + self.n_slices
        if self.ax ==0:
            new_img[start:end] = 0
        elif self.ax ==1:
            new_img[:,start:end] = 0
        elif self.ax ==2:
            new_img[:,:,start:end] = 0
        return new_img
    
class RandomMotionAdapted2D(RandomMotion):
    """
    Adapted to not sample from uniform distribution but choose from a list of values and fix rotation and translation axis for 2D implementation
    """

    def get_params(
        self,
        degrees_range: Tuple[float, float],
        translation_range: Tuple[float, float],
        num_transforms: int,
        perturbation: float = 0.3,
        is_2d: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # If perturbation is 0, time intervals between movements are constant

        rotation_axis = 2
        degrees_params = self.get_params_array_custom(
            degrees_range,
            num_transforms,
            rotation_axis,
        )
        translation_axis = random.choice([0, 1])
        translation_params = self.get_params_array_custom(
            translation_range,
            num_transforms,
            translation_axis,
        )
        step = 1 / (num_transforms + 1)
        times = torch.arange(0, 1, step)[1:]
        noise = torch.FloatTensor(num_transforms)
        noise.uniform_(-step * perturbation, step * perturbation)
        times += noise
        times_params = times.numpy()
        return times_params, degrees_params, translation_params

    @staticmethod
    def get_params_array_custom(nums_range: Tuple[float, float], num_transforms: int, axis: int):

        tensor = torch.FloatTensor(num_transforms, 3)
        for i in range(num_transforms):
            tensor[i] = torch.FloatTensor([random.choice(nums_range) if j == axis else 0 for j in range(3)])
        return tensor.numpy()



class RandomMotionAdapted(RandomMotion):
    """
    Adapted to not sample from uniform distribution but choose from a list of values
    """

    def get_params(
        self,
        degrees_range: Tuple[float, float],
        translation_range: Tuple[float, float],
        num_transforms: int,
        perturbation: float = 0.3,
        is_2d: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # If perturbation is 0, time intervals between movements are constant

        rotation_axis = 2
        degrees_params = self.get_params_array_custom(
            degrees_range,
            num_transforms,
            rotation_axis,
        )
        translation_axis = random.choice([0, 1, 2])
        translation_params = self.get_params_array_custom(
            translation_range,
            num_transforms,
            translation_axis,
        )
        if is_2d:  # imagine sagittal (1, A, S)
            degrees_params[:, :-1] = 0  # rotate around Z axis only
            translation_params[:, 2] = 0  # translate in XY plane only
        step = 1 / (num_transforms + 1)
        times = torch.arange(0, 1, step)[1:]
        noise = torch.FloatTensor(num_transforms)
        noise.uniform_(-step * perturbation, step * perturbation)
        times += noise
        times_params = times.numpy()
        return times_params, degrees_params, translation_params

    @staticmethod
    def get_params_array_custom(nums_range: Tuple[float, float], num_transforms: int, axis: int):

        tensor = torch.FloatTensor(num_transforms, 3)
        for i in range(num_transforms):
            tensor[i] = torch.FloatTensor([random.choice(nums_range) if j == axis else 0 for j in range(3)])
        return tensor.numpy()


class RandomSpikeAdapted(RandomSpike):
    """
    Adapted to not sample intensity from uniform distribution but choose from a list of values
    """
    def get_params(self, num_spikes_range: Tuple[int, int], intensity_range: Tuple[float, float],
    ) -> Tuple[np.ndarray, float]:
        ns_min, ns_max = num_spikes_range
        num_spikes_param = int(torch.randint(ns_min, ns_max + 1, (1,)).item())
        intensity_param = random.choice(intensity_range)
        spikes_positions = torch.rand(num_spikes_param, 3).numpy()
        return spikes_positions, intensity_param
    
import matplotlib.pyplot as plt

from torchio.transforms.augmentation.intensity.random_swap import get_random_indices_from_shape
class RandomSwapAdapted(RandomSwap):

    @staticmethod
    def get_params(
        tensor: torch.Tensor,
        patch_size: np.ndarray,
        num_iterations: int,
    ):
        si, sj, sk = tensor.shape[-3:]
        
        
        spatial_shape = si, sj, sk  # for mypy
        locations = []
        for _ in range(num_iterations):
            while True:
                first_ini, first_fin = get_random_indices_from_shape(
                    spatial_shape,
                    patch_size.tolist(),
                )
                patch1 = tensor[0][first_ini[0]:first_fin[0], first_ini[1]:first_fin[1], first_ini[2]:first_fin[2]]
                patch1_foreground_perc = torch.where(patch1 > 0, 1, 0).sum()/patch1.numel()
                
                if patch1_foreground_perc < 0.1:
                    continue
                else:
                    break

            while True:
                second_ini, second_fin = get_random_indices_from_shape(
                    spatial_shape,
                    patch_size.tolist(),
                )
                larger_than_initial = np.all(second_ini >= first_ini)
                less_than_final = np.all(second_fin <= first_fin)
            
                patch2 = tensor[0][second_ini[0]:second_fin[0], second_ini[1]:second_fin[1], second_ini[2]:second_fin[2]]
                patch2_foreground_perc = torch.where(patch2 > 0, 1, 0).sum()/patch2.numel()

                if larger_than_initial and less_than_final:
                    continue  # patches overlap
                elif patch2_foreground_perc < 0.1:
                    continue  # to far outside body contour
                else:
                    break  # patches don't overlap
            location = tuple(first_ini), tuple(second_ini)
            locations.append(location)


        return locations  # type: ignore[return-value]
    

