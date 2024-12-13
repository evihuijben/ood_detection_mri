import argparse


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Base Options")

    def initialize(self):
        self.parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory to save checkpoints.")
        self.parser.add_argument("--seed", type=int, default=42, help="seed for reproducibility")

        # Data and dataloader parameters
        self.parser.add_argument("--total_slices", type=int, default=256, help="total number of slices in the data set.")
        self.parser.add_argument("--which_slices", type=str, default='all', help="which slices to use, 'all', 'mid', 'mid_x'")
        self.parser.add_argument("--is_grayscale", type=int, default=1, help="Is data grayscale.")
        self.parser.add_argument("--spatial_dimension", type=int, default=2, help="spatial dimension of images.")
        self.parser.add_argument("--batch_size", type=int, default=4, help="Training batch size.")
        self.parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
        

        # General model parameters
        self.parser.add_argument("--vae_name", required=True , help="Name of your model. For inference it is the folder where the VAE is saved: YYMMDD_HHMMSS_<given_name>.")
        self.parser.add_argument("--latent_channels", type=int, default=32,)
        self.parser.add_argument("--vae_norm_num_groups", type=int, default=32,)
        self.parser.add_argument("--vae_num_res_blocks", type=int, default=2,)
        self.parser.add_argument("--vae_attention_levels", type=str, default="0,0,0,0,0,1",)
        self.parser.add_argument("--vae_num_channels", type=str, default="128,128,256,256,512,512")
        
        # Loss weights
        self.parser.add_argument('--adv_weight', type=float, default=0.005, help="Weight for adversarial loss.")
        self.parser.add_argument('--perceptual_weight', type=float, default=0.002, help="Weight for perceptual loss.")
        self.parser.add_argument('--kl_weight', type=float, default=1e-8, help="Weight for KL divergence loss.")



    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()

        if self.opt.spatial_dimension != 2:
            raise NotImplementedError("Only 2D data is supported at the moment.")

        self.opt.vae_attention_levels = [bool(int(x)) for x in self.opt.vae_attention_levels.split(',')]
        self.opt.vae_num_channels = [int(x) for x in self.opt.vae_num_channels.split(',')]

        if self.phase == 'inference':
            self.opt.metric_list = self.opt.metric_list.split(',')

        return self.opt


class TrainOptions(BaseOptions):
    def initialize(self):
        self.phase = "train"
        super().initialize()

        

        self.parser.add_argument("--data_dir_train", type=str, required=True, help="Training data directory.")
        self.parser.add_argument("--data_dir_val", type=str, required=True, help="Validation data directory.")

        self.parser.add_argument('--n_epochs', type=int, default=100, help="Number of epochs to train.")

        # Learning rates
        self.parser.add_argument("--lr_G", type=float, default=1e-4, help="Learning rate for generator.")
        self.parser.add_argument("--lr_D", type=float, default=5e-4, help="Learning rate for discriminator.")

        # Wandb parameters
        self.parser.add_argument("--wandb_entity", type=str, default="", help="wandb entity")
        self.parser.add_argument("--wandb_project", type=str, default="", help="project name")
        
        self.initialized = True

class InferenceOptions(BaseOptions):
    def initialize(self):
        self.phase = "inference"
        super().initialize()

        

        self.parser.add_argument("--data_dir_inference", type=str, required=True, help="Inference data directory.")
        self.parser.add_argument("--results_folder", type=str, required=True, help="Path to root folder for the results")

        self.parser.add_argument("--vae_epoch", type=str, default="last" , help="epoch of the vae model for loading. ['best', 'last']")
        self.parser.add_argument("--metric_list", default='MAE,SSIM,LPIPS', help="List of metrics to calculate.")
        
        self.parser.add_argument("--save_recons", action='store_true', help="Save reconstructions.")
        self.parser.add_argument("--save_snapshots", action='store_true', help="Save snapshots.")
        self.parser.add_argument("--save_predictions", action='store_true', help="Save predictions.")


        self.initialized = True

