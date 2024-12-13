
#%%
import numpy as np
import torchio as tio
import pandas as pd
import json
import nibabel as nib
import custom_transforms
import os
from custom_transforms import set_seed, define_kwargs



def create_local_transformations(src, dst, types):
    dst_label_pixel = dst + '_label'
    os.makedirs(dst, exist_ok=True)
    os.makedirs(dst_label_pixel, exist_ok=True)

    
    all_files = sorted(os.listdir(src))
    n_files = len(all_files)



    kwarg_options = define_kwargs(types, n_files)

    details = []
    for file_i, file in enumerate(all_files):
        print(file_i, file)
        subject = tio.Subject(image=tio.ScalarImage(os.path.join(src, file)))
        for augm_type, kwargs_selected in kwarg_options.items():
            # Define details
            kwargs = {k: v[file_i] for k, v in kwargs_selected.items()}
            print('\t', augm_type, kwargs)
            new_fname = file.split('.')[0] + f'_{augm_type}.nii.gz'
            row_df = pd.DataFrame([{'original_filename': file, 'aug_type': augm_type ,'new_filename': new_fname, 'kwargs': json.dumps(kwargs, default=int)}])
            details.append(row_df)
            
            this_transform = getattr(custom_transforms, augm_type)(**kwargs)
            transformed_subject = this_transform(subject)
        
            img2save = nib.Nifti1Image( transformed_subject['image'].data.squeeze().numpy(), transformed_subject['image'].affine)
            nib.save(img2save, os.path.join(dst, new_fname))
        
            # Create a new subject with the local anomaly and save to  os.path.join(dst_label_pixel, new_fname)
            local_anomaly = np.where(transformed_subject['image'].data.squeeze().numpy()== subject['image'].data.squeeze().numpy(), 0, 1)
            local_anomaly_img = nib.Nifti1Image(local_anomaly.astype(np.uint8), subject['image'].affine)
            nib.save(local_anomaly_img, os.path.join(dst_label_pixel, new_fname))            

    df = pd.concat(details, axis=0)
    df.to_csv(dst + '_local.csv', index=False)
    print('Details saved to', dst + '_local.csv')



def create_global_transformations(src, dst, types, which_slices='all', isize=256):  
    os.makedirs(dst, exist_ok=True)

    if which_slices.startswith('mid'):
        n_included = min([int(which_slices.split('_')[-1]), isize])
        print("Using only the middle", n_included, "slices")
        start_slice = (isize - n_included) // 2
        end_slice = start_slice + n_included
        included_slices = range(start_slice, end_slice)
    elif which_slices == 'all':
        included_slices = range(isize)
        n_included = isize
    else:
        raise ValueError(f"which_slices should be 'all' or 'mid_x' with x an integer, not '{which_slices}'")

    all_files = sorted(os.listdir(src))
    n_files = len(all_files) * n_included
    kwarg_options = define_kwargs(types, n_files)


    details = []
    file_slice_i = -1
    for file_i, file in enumerate(all_files):
        
        subject = tio.Subject(image=tio.ScalarImage(os.path.join(src, file)))
        normalize = tio.RescaleIntensity((0, 1))
        subject = normalize(subject)
        for slice_i in included_slices:
            file_slice_i += 1

            print(file_i, file, 'slice', slice_i)

            for augm_type, kwargs_selected in kwarg_options.items(): 
                kwargs = {k: v[file_slice_i] for k, v in kwargs_selected.items()}
                new_fname =  f"{file.split('.')[0]}_{slice_i}_{augm_type}.nii.gz"


                row_df = pd.DataFrame([{'original_filename': file, 'aug_type': augm_type ,'new_filename': new_fname, 'kwargs': json.dumps(kwargs, default=int)}])
                details.append(row_df)
                print('\t', augm_type, kwargs)

                this_transform = getattr(custom_transforms, augm_type)(**kwargs)
                transformed_subject = this_transform(subject)
                transformed_subject = normalize(transformed_subject)
                
                img2save = nib.Nifti1Image(transformed_subject['image'].data.squeeze().numpy()[...,slice_i], transformed_subject['image'].affine)
                nib.save(img2save, os.path.join(dst, new_fname))

    df = pd.concat(details, axis=0)
    df.to_csv(dst + '_global.csv', index=False)
    print('Details saved to', dst + '_global.csv')



if __name__ == '__main__':
    print('started')
    set_seed(0)

    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='.', help='data directory')
    parser.add_argument('--which_dataset', type=str, help="which dataset to create, either 'toy' or 'transformed'")
    parser.add_argument('--global_transforms', action='store_true', help="If the transformations are global")
    parser.add_argument('--which_slices', type=str, default='mid_20', help="which slices to save for global transformations, either 'all' or 'mid_x' with x an integer")
    args = parser.parse_args()


    if args.global_transforms == True:
        src = os.path.join(args.dataroot, 'brain_val', "brain_val")
    else:
        src = os.path.join(args.dataroot, 'brain_val', "brain_val_normalized_2D")
    dst = src + f'_{args.which_dataset}'
    

    
    if args.which_dataset == 'toy':
        if args.global_transforms == True:
            raise ValueError("Toy dataset consists of local transformations only")

        print('Creating toy dataset (local)')
        

        types = [       
            ('Circle2D', {'radius': np.random.choice(range(20, 41), size=80),
                        'intensity': np.random.uniform(0, 1, size=80),
                        'smoothing': [False]
                        }),
        ]

        create_local_transformations(src, dst, types,)

    elif args.which_dataset == 'transformed':
        n_groups = 10

        if args.global_transforms == False:
            print('Creating transformed dataset (local)')
                
            local_types = [

                ('BlackSlice', {'ax': np.array([0,1]), 
                                'n_slices': [1,2,3,4,5]}),

                ('Circle2D', {'radius': np.linspace(3,30,n_groups).astype(int), 
                            'intensity': np.linspace(0, 1, n_groups)}),

                ('RandomSwapAdapted', {'patch_size': [(x,x,1) for x in np.linspace(30,75,n_groups).astype(int)], 
                                'num_iterations': 1,}), 
            ]
            create_local_transformations(src, dst, local_types)
        elif args.global_transforms == True:
            print('Creating transformed dataset (global)')

            global_types = [

            ('RandomBlur', {'std': [(x,x,) for x in np.linspace(0.25, 2.5, n_groups)],}),

            ('RandomBiasField', {'coefficients': [(x,x,) for x in np.linspace(0.05, 0.5, n_groups)],}),

            ('RandomMotionAdapted2D', {'degrees': [(-x,x,) for x in np.linspace(1, 10, n_groups)], 
                                'translation': [(-x,x,) for x in np.linspace(1.5, 15, n_groups)],
                                'num_transforms': 1,}),

            ('RandomGhosting', { 'num_ghosts': [(x,x,) for x in [1,2]],
                                'intensity': [(x,x,) for x in np.linspace(0.4, 0.6, n_groups)],
                                'axes': [0,1],}),

            ('RandomSpikeAdapted', {'num_spikes': [(x,x,) for x in [1,2]], 
                                    'intensity': [(-x,x,) for x in np.linspace(0.25, 2.5, n_groups)],}),

            ('RandomNoise', {'std': [(x,x,) for x in np.linspace(0.01,0.37,n_groups)],}),

            ('RandomElasticDeformation', {'num_control_points': [5,6,7,8], 
                                        'max_displacement': np.linspace(7.5, 30, n_groups), }),


            ]


            create_global_transformations(src, dst, global_types, args.which_slices)

    else:
        raise ValueError(f"which_dataset should be 'toy' or 'transformed', not '{args.which_dataset}'")
    
    print('Finished')


        
     
