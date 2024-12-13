#%%
import os
import pandas as pd
import numpy as np
import torch
import nibabel as nib
from pathlib import Path
import time
import argparse
HIGHER_BETTER = ['SSIM', 'luminance', 'contrast', 'structure']



def compute_classification_outcomes(y_true, y_score, outcomes):
    y_true = y_true.flatten()
    y_score = y_score.flatten()
   
    
    for th_i, th in enumerate(outcomes.keys()):
        y_pred = (y_score >= th).type(torch.int8)
        
        outcomes[th]['TP'] += ((y_pred == 1) & (y_true == 1)).sum().item()
        outcomes[th]['FP'] += ((y_pred == 1) & (y_true == 0)).sum().item()
        outcomes[th]['FN'] += ((y_pred == 0) & (y_true == 1)).sum().item()
        outcomes[th]['TN'] += ((y_pred == 0) & (y_true == 0)).sum().item()
        
    return outcomes

# Function to calculate AP for a specific metric
def calculate_auc(df):

    # Sort by Recall
    df = df.sort_values(by='Recall')
    # Calculate Average Precision (AP) using trapezoidal rule
    return np.trapz(df['Precision'], df['Recall'])

def calculate_auprc(df):

    df_grouped = df.groupby(['model_names', 'dataset', 'vae_epoch', 'metric_list', 'threshold']).sum().reset_index()

    # Precision: Set to 1 when (TP + FP) == 0
    df_grouped['Precision'] = np.where(
        (df_grouped['TP'] + df_grouped['FP']) == 0, 
        1, 
        df_grouped['TP'] / (df_grouped['TP'] + df_grouped['FP'])
    )

    # Recall: Set to 0 when (TP + FN) == 0, otherwise calculate normally
    df_grouped['Recall'] = np.where(
        (df_grouped['TP'] + df_grouped['FN']) == 0, 
        0, 
        df_grouped['TP'] / (df_grouped['TP'] + df_grouped['FN'])
    )

    # F1 Score: Using Precision and Recall, set to 0 when (Precision + Recall) == 0
    df_grouped['F1'] = np.where(
        (df_grouped['Precision'] + df_grouped['Recall']) == 0, 
        0, 
        2 * (df_grouped['Precision'] * df_grouped['Recall']) / (df_grouped['Precision'] + df_grouped['Recall'])
    )
    
    # Sort by Recall
    df_AUC = df_grouped.groupby(['model_names', 'dataset', 'vae_epoch', 'metric_list']).apply(calculate_auc)
    df_AUC = df_AUC.reset_index()
    df_AUC.columns = ['model_names', 'dataset', 'vae_epoch', 'metric_list', 'AUPRC']
    return df_AUC




def calculate_results(args):
    thresholds = np.linspace(0, 1, num=51)
    predictors = ['TP', 'FP', 'FN', 'TN']



    final_outcomes = {th: {p: 0 for p in predictors}  for th in thresholds }

    dataset = Path(args.data_dir).name


    # check if all reconstructions are the same
    all_ls = [sorted(os.listdir(os.path.join(args.results_folder, model_name, dataset, f"epoch{args.vae_epoch}", 'recon'))) for model_name in args.model_names.split(',')]
    if not all([ls == all_ls[0] for ls in all_ls]):
        raise ValueError("Not all reconstructions are the same.")

    recons = all_ls[0]
    print(len(recons))
    for fname in recons:
        print(f"Processing {fname}")
        label_path = os.path.join(args.data_dir + "_label", fname)
        if os.path.exists(label_path):
            label_img = nib.load(label_path).get_fdata()
        else:
            raise ValueError(f"Label file {label_path} not found.")

        label_img = torch.from_numpy(label_img)

        all_maps = []
        for model_name in args.model_names.split(','):
            maps = torch.load(os.path.join(args.results_folder, model_name, dataset, f"epoch{args.vae_epoch}", 'predictions', f"{fname}.pt"))
            for m in args.metric_list.split(','):
                this_map = maps[m]

                ##### Normalize maps
                this_map = (1-this_map) if m in HIGHER_BETTER else this_map 
                all_maps.append(this_map)

        avg_map = np.mean(all_maps, axis=0)
        avg_map = torch.from_numpy(avg_map)

        # Compute TP, FP, TN, FN
        final_outcomes = compute_classification_outcomes(label_img, avg_map, final_outcomes)

    all_accumulated = []
    for th in final_outcomes.keys():
        this_dict = {'model_names': args.model_names,
                     'dataset': dataset,
                     'vae_epoch': args.vae_epoch,
                     'metric_list': args.metric_list,
                     'threshold': th}
        for p in predictors:
            this_dict[p] = final_outcomes[th][p]
        all_accumulated.append(this_dict)

    all_accumulated = pd.DataFrame(all_accumulated)
    date_str = time.strftime('%y%m%d_%H%M%S')
    csv_file = os.path.join(args.results_folder, f"Classification_outcomes_{date_str}.csv")
    all_accumulated.to_csv(csv_file, index=False)
    print(f"Predictors saved to {csv_file}")
    

    df_AUC = calculate_auprc(all_accumulated)
    csv_file = os.path.join(args.results_folder, f"PRAUC_{date_str}.csv")
    df_AUC.to_csv(csv_file, index=False)
    print(f"AUC saved to {csv_file}")





if __name__ == '__main__':
    print('Started')
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_names", type=str, required=True, help="Comma separated list of model names.")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory.")
    parser.add_argument("--results_folder", type=str, required=True, help="Results directory.")
    parser.add_argument("--vae_epoch", type=str, default="last", help="epoch of the vae model for loading. ['best', 'last']")
    parser.add_argument("--metric_list", default='contrast,LPIPS', help="List of metrics to include ['MAE', 'SSIM', 'LPIPS', 'contrast', 'structure', 'luminance'].")
   
    args = parser.parse_args()

    calculate_results(args)


