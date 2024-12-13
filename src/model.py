import torch
from monai.networks.layers import Act
from torch.nn import L1Loss
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator
import os


class AutoEncoderNoKL(AutoencoderKL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_checkpointing:
            h = torch.utils.checkpoint.checkpoint(self.encoder, x, use_reentrant=False)
        else:
            h = self.encoder(x)

        z = self.quant_conv_mu(h)
        return z, None


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, _ = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z, None

    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.encode(x)
        return z



def define_AE(args, device, discriminator=True):
    print("#"*50)
    print("define the networks".center(50))
    print("#"*50)

    if args.kl_weight == 0:
        ae = AutoEncoderNoKL
    else:
        ae = AutoencoderKL

    model = ae(
        spatial_dims=args.spatial_dimension,
        in_channels=1 if args.is_grayscale else 3,
        out_channels=1 if args.is_grayscale else 3,
        num_channels=args.vae_num_channels,
        latent_channels=args.latent_channels,
        num_res_blocks=args.vae_num_res_blocks,
        norm_num_groups=args.vae_norm_num_groups,
        attention_levels=args.vae_attention_levels,
    )
    model.to(device)
    print(f"{sum(p.numel() for p in model.parameters()):,} vae_model parameters")

    if discriminator == False:
        return model
    else:
        discriminator = PatchDiscriminator(
            spatial_dims=args.spatial_dimension,
            num_layers_d=3,
            num_channels=64,
            in_channels=1 if args.is_grayscale else 3,
            out_channels=1 if args.is_grayscale else 3,
            kernel_size=4,
            activation=(Act.LEAKYRELU, {"negative_slope": 0.2}),
            norm="BATCH",
            bias=False,
            padding=1,
        )
        discriminator.to(device)

        return model, discriminator

    


def save_checkpoint(path, epoch, model):
    checkpoint = {
        "epoch": epoch + 1, 
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, path)



def save_checkpoints(generator, discriminator, run_dir, epoch , suffix):
    save_checkpoint(os.path.join(run_dir, f"checkpointG_{suffix}.pth"), epoch, generator) 
    save_checkpoint(os.path.join(run_dir, f"checkpointD_{suffix}.pth"), epoch, discriminator)
    



class myLosses():
    def __init__(self, args, device):
        self.perceptual_loss = PerceptualLoss(spatial_dims=args.spatial_dimension, network_type="vgg")
        self.perceptual_loss.to(device)
        self.l1_loss = L1Loss()
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.args = args

    def calculate_generator_losses(self, reconstruction, images, z_mu, z_sigma, logits_fake=None):

        recons_loss = self.l1_loss(reconstruction.float(), images.float())

        if self.args.kl_weight == 0:
            kl_loss = torch.tensor(0.0).to(reconstruction.device)
        else:
            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            kl_loss = self.args.kl_weight * kl_loss

        p_loss =  self.args.perceptual_weight *  self.perceptual_loss(reconstruction.float(), images.float())
        if logits_fake is None:
            adv_loss = torch.tensor(0.0).to(reconstruction.device)
        else:
            adv_loss = self.args.adv_weight * self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)


        losses_dict =   {
                "recons_loss": recons_loss,
                "gen_adv_loss": adv_loss,
                "pereceptual_loss": p_loss,
                "KL_loss": kl_loss,
                "gen_loss_all": recons_loss + kl_loss +p_loss + adv_loss,
            }
        
        return losses_dict

    def calculate_discriminator_losses(self, logits_fake, logits_real):
        loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
        loss_d = self.args.adv_weight * discriminator_loss

        return {"discr_loss": loss_d}