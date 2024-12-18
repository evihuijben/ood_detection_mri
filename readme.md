# Repository OOD detection
This repository provides the code and resources to recreate the dataset and reproduce the results from the paper: Huijben et al. (2025) "Ensembling Reconstruction-Based Autoencoders for Out-of-Distribution Detection in Brain MRI: A Metric-Driven Approach"


## Setup
Install python requirements

```
pip install -r requirements.txt
```


## Dataset preparation
We used the publicly available brain MRI dataset from the [MOOD challenge](https://www.synapse.org/Synapse:syn21343101/wiki/599515).

### Step 1: Dataset split
Split the MOOD brain dataset into training and validation according to the split in `dataset/MOOD_brain_split.csv`. Structure the files as follows:


```
<dataroot>
  ├── brain_train
  │   ├── brain_train
  │   │   └── <id>.nii.gz
  │   │   └── ... 
  ├── brain_val
  │   ├── brain_val
  │   │   └── <id>.nii.gz
  │   │   └── ... 
```

### Step 2: Normalize and save as 2D slices
Normalize the 3D volumes, save all mid 20 (and non-zero) 2D slices separately by running:

```
python ./dataset/normalize_and_save_2D.py --dataroot <dataroot> --which_slices mid_20
```

The resulting files are structures as follows:
```
<dataroot>
  ├── brain_train
  │   ├── brain_train_normalized_2D
  │   │   └── <id>_<slice>.nii.gz
  │   │   └── ... 
  ├── brain_val
  │   ├── brain_val_normalized_2D
  │   │   └── <id>_<slice>.nii.gz
  │   │   └── ... 
```

These folders are used for training (`brain_train_normalized_2D`) and for selecting the optimal epoch (based on `brain_val_normalized_2D`).

### Step 3: Create the MOOD<sub>val,toy</sub> dataset.
To create MOOD<sub>val,toy</sub> run the following:

```
python ./dataset/transform_val_set.py --dataroot <dataroot> --which_dataset toy
```

The transformed images and the ground turth labels are saves as follows:

```
<dataroot>  
  ├── brain_val
  │   ├── brain_val_normalized_2D_toy
  │   │   └── <id>_<slice>_Cicle2D.nii.gz
  │   │   └── ...
  │   ├── brain_val_normalized_2D_toy_label
  │   │   └── <id>_<slice>_Cicle2D.nii.gz
  │   │   └── ... 
```


### Step 4: Create the MOOD<sub>transformed</sub> dataset.
To create the local and global transformations of MOOD<sub>transformed</sub> run the following two lines:

```
python ./dataset/transform_val_set.py --dataroot <dataroot> --which_dataset transformed
```

```
python ./dataset/transform_val_set.py --dataroot <dataroot> --which_dataset transformed --global
```

The dataset, consisting of the subsets toy lesion (`Circle2D`), black stripe (`BlackSlice`), patch swap (`RandomSwapAdapted`), blur (`RandomBlur`), noise (`RandomNoise`), elastic deformation (`RandomElasticDeformation`), motion (`RandomMotionAdapted2D`), bias field (`RandomBiasField`), ghosting (`RandhomGhosting`), and spike (`RandomSpikeAdapted`) are saved as follows: 

```
<dataroot>  
  ├── brain_val
  │   ├── brain_val_normalized_2D_transformed
  │   │   └── <id>_<slice>_Cicle2D.nii.gz
  │   │   └── <id>_<slice>_BlackSlice.nii.gz
  │   │   └── <id>_<slice>_RandomSwapAdapted.nii.gz
  │   │   └── <id>_<slice>_RandomBlur.nii.gz
  │   │   └── <id>_<slice>_RandomNoise.nii.gz
  │   │   └── <id>_<slice>_RandomElasticDeformation.nii.gz
  │   │   └── <id>_<slice>_RandomMotionAdapted2D.nii.gz
  │   │   └── <id>_<slice>_RandomBiasField.nii.gz
  │   │   └── <id>_<slice>_RandomGhosting.nii.gz
  │   │   └── <id>_<slice>_RandomSpikeAdapted.nii.gz
  │   │   └── ...
  │   ├── brain_val_normalized_2D_transformed_label
  │   │   └── <id>_<slice>_Cicle2D.nii.gz
  │   │   └── <id>_<slice>_BlackSlice.nii.gz
  │   │   └── <id>_<slice>_RandomSwapAdapted.nii.gz
  │   │   └── ... 
```


## Training
To train the model as presented in the paper, first define the model parameters:

```
model2x2_flags="--kl_weight 0 --vae_attention_levels 0,0,0,0,0,0,0,1 --vae_num_channels 128,128,256,256,512,512,512,512"

model4x4_flags="--kl_weight 0 --vae_attention_levels 0,0,0,0,0,0,1 --vae_num_channels 128,128,256,256,512,512,512"

model8x8_flags="--kl_weight 0 --vae_attention_levels 0,0,0,0,0,1 --vae_num_channels 128,128,256,256,512,512"

model16x16_flags="--kl_weight 0 --vae_attention_levels 0,0,0,0,1 --vae_num_channels 128,128,256,256,512"
```

Then define your dataset directory and checkpoint directory (where the models will be saved) as follows:
```
train_dir_flags="--data_dir_train <dataroot>/brain_train/brain_train_normalized_2D --data_dir_val <dataroot>/brain_val/brain_val_normalized_2D --checkpoint_dir <checkpoint_dir>"
```

There is an option to log your training using [Weights & Biases](https://wandb.ai/home) using the following flags. If you don't use them, the progress is not logged in Weights & Biases, but training is not affected.
```
wandb_flags="--wandb_entity <wandb_entity> --wandb_project <wandb_project>"
```

Train the model as follows, but note that:
* `model_flags` should be replaced with the desired model: `model2x2_flags`, `model4x4_flags`, `model8x8_flags`, or `model16x16_flags`
* The dataset folder now contains only the mid 20 slices, but in case that needs to be adjusted on the fly, use `--which_slices mid_20`
```
python ./train.py $model_flags $train_dir_flags $wandb_flags --vae_name <given_name> 
```


## Inference
Define the paths for the dataset to be tested, the checkpoint directory and the folder where the results are stored.
```
inf_dir_flags="--data_dir_inference <dataroot>/brain_val/brain_val_normalized_2D_toy --checkpoint_dir  <checkpoint_dir> --results_folder <results_folder>"
```

Define what needs to be saved. `--save_recons` saves the reconstructions, `--save_snapshots` saves snapshots of the prediction maps for all metrics, and `--save_predictions` saves all the prediction maps.
```
saving_flags="--save_recons --save_snapshots --save_predictions"
```

Run the inference code as follows. *Again select the correct model flags as identified during training.*

```
python ./inference.py $model_flags $inf_dir_flags $saving_flags --vae_name <YYMMDD_HHMMSS_given_name>
```


## Calculate AUPRC
After saving the prediction maps during inference, the AUPRC can be calculated, given the anomalies are local.

```
python ./calculate_auprc.py --model_names <comma_separated_model_names> --data_dir <dataroot>/brain_val/brain_val_normalized_2D_toy --results_folder <results_folder> --vae_epoch last --metric_list "contrast,LPIPS"
```
