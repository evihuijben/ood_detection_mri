import os
import nibabel as nib
import numpy as np
from tqdm import tqdm


def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image


def normalize_and_save_slices(src, which_slices='all', isize=256):
    dst = src + '_normalized_2D'
    os.makedirs(dst, exist_ok=True)


    if which_slices.startswith('mid'):
        n_included = min([int(which_slices.split('_')[-1]), isize])
        start_slice = (isize - n_included) // 2
        end_slice = start_slice + n_included
        included_slices = range(start_slice, end_slice)
    elif which_slices == 'all':
        included_slices = range(isize)
    else:
        raise ValueError(f"which_slices should be 'all' or 'mid_x' with x an integer, not '{which_slices}'")

    for fname in tqdm(set(included_slices).intersection(img_data.sum((0,1)).nonzero()[0])):

    
        img_path = os.path.join(src, fname)
        img = nib.load(img_path)
        img_data = img.get_fdata()
        img_data = normalize_image(img_data)


        for slice_i in img_data.sum((0,1)).nonzero()[0]:


            nib_slice = nib.Nifti1Image(img_data[...,slice_i], affine=img.affine) 
            nib.save(nib_slice, os.path.join(dst, f"{fname.split('.')[0]}_{slice_i:03d}.nii.gz"))



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='.', help='data directory')
    parser.add_argument('--which_slices', type=str, default='mid_20', help="which slices to save, either 'all' or 'mid_x' with x an integer")
    args = parser.parse_args()

    src_train = os.path.join(args.dataroot, 'brain_train', "brain_train")
    normalize_and_save_slices(src_train)
    
    src_val = os.path.join(args.dataroot, 'brain_val', "brain_val")
    normalize_and_save_slices(src_val)

