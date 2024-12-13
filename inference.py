#%%
import os
import torch
from monai.utils import set_determinism
import torch
from pathlib import Path
import nibabel as nib
import pandas as pd

from src.data import get_data_loader
from src.model import define_AE
from src.options import InferenceOptions
from src.postprocessing_utils import calculate_merics_and_snapshot2D, CustomMetrics


##############
args = InferenceOptions().parse()


set_determinism(seed=args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed) # if use multi-GPU

print("#"*50)
print("loading data".center(50))
print("#"*50 + "\n")
val_loader = get_data_loader(args.data_dir_inference ,args, shuffle=False )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
                                  
vae_model = define_AE(args, device, discriminator=False)
checkpoint_path= os.path.join(args.checkpoint_dir, args.vae_name ,"checkpointG_{}.pth".format(args.vae_epoch))
checkpoint = torch.load(checkpoint_path, map_location=device)
vae_model.load_state_dict(checkpoint["model_state_dict"])
print(f"Loaded VAE checkpoint: {checkpoint_path}")
vae_model.eval()

saveto = os.path.join(args.results_folder, 
                      args.vae_name, 
                      Path(args.data_dir_inference).name,
                      f"epoch{args.vae_epoch}", 
                      )





os.makedirs(saveto, exist_ok=True)
if args.save_recons:
    recons_folder = os.path.join(saveto, 'recon')
    os.makedirs(recons_folder, exist_ok=True)
    print("Reconstructions being saved to", recons_folder)

if args.save_snapshots:
    snapshots_folder = os.path.join(saveto, f"snapshots")
    os.makedirs(snapshots_folder, exist_ok=True)
    print("Snapshots being saved to", snapshots_folder)
else:
    snapshots_folder = None

if args.save_predictions:
    predictions_folder = os.path.join(saveto, 'predictions')
    os.makedirs(predictions_folder, exist_ok=True)
    print("Predictions being saved to", predictions_folder)



#%%
my_metrics = CustomMetrics(args.metric_list)
   

print()
print('Processing inference data ...')
all_results = []
with torch.no_grad():
    for val_step, batch in enumerate(val_loader):
        if val_step%500 == 0:
            print(f"Inference step {val_step}/{len(val_loader)}")

        z_mu, _ = vae_model.encode(batch["image"].float().to(device))
        reconstruction_batch = vae_model.decode(z_mu).detach().cpu()

        if args.save_recons:
            for image, recon, path in zip(batch["image"], reconstruction_batch, batch["path"]):
                nifti_recon = nib.Nifti1Image(recon.squeeze().numpy(), affine=image.affine)
                nib.save(nifti_recon, os.path.join(recons_folder, Path(path).name))


        patient_results, patient_maps = calculate_merics_and_snapshot2D(batch["path"],
                                                                my_metrics,
                                                                batch["image"][:,0],
                                                                reconstruction_batch[:,0], 
                                                                snapshots_folder)
        
        if args.save_predictions:            
            for i, p in enumerate(batch['path']):
                one_patient_maps = {k: v[i] for k, v in patient_maps.items()}
                torch.save(one_patient_maps, os.path.join(predictions_folder, Path(p).name + ".pt"))

        all_results.extend(patient_results)


df = pd.DataFrame(all_results)
df.to_csv(os.path.join(saveto, f"Recon_errors_avg.csv"), index=False)
print()
print("Results saved to", os.path.join(saveto, f"Recon_errors_avg.csv"))
print('finished')

