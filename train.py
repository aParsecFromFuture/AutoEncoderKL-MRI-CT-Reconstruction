import os
import glob
import random

import wandb

import torch
from torchvision.utils import make_grid
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
from tqdm import tqdm

import monai
from monai import transforms
from monai.data import CacheDataset, DataLoader
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator

def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
    return torch.sum(kl_loss) / kl_loss.shape[0]


n_epochs = 2000
batch_size = 4

autoencoder_warm_up_n_epochs = 10
min_val_loss = 1e6
val_interval = 10

adv_weight = 0.01
perceptual_weight = 0.001
kl_weight = 1e-6

train_dir = 'dataset/train'
val_dir = 'dataset/val'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

data_transform = transforms.Compose([
    transforms.LoadImaged(keys=['mr', 'ct'], image_only=True),
    transforms.EnsureChannelFirstd(keys=['mr', 'ct']),
    transforms.EnsureTyped(keys=['mr', 'ct']),
    transforms.Orientationd(keys=['mr', 'ct'], axcodes='RAS'),
    transforms.ScaleIntensityRanged(keys=['ct'], a_min=-1024, a_max=3000, b_min=0, b_max=1),
    transforms.ScaleIntensityd(keys=['mr'], minv=0, maxv=1),
    transforms.RandSpatialCropd(keys=['mr', 'ct'], roi_size=(64, 64, 64), random_size=False),
])

train_mr_images = sorted(glob.glob(os.path.join(train_dir, '*', 'mr.nii.gz')))
train_ct_images = sorted(glob.glob(os.path.join(train_dir, '*', 'ct.nii.gz')))

train_dicts = [{'mr': mr_image, 'ct': ct_image}
               for mr_image, ct_image in zip(train_mr_images, train_ct_images)]

val_mr_images = sorted(glob.glob(os.path.join(val_dir, '*', 'mr.nii.gz')))
val_ct_images = sorted(glob.glob(os.path.join(val_dir, '*', 'ct.nii.gz')))

val_dicts = [{'mr': mr_image, 'ct': ct_image}
             for mr_image, ct_image in zip(val_mr_images, val_ct_images)]

train_ds = CacheDataset(
    data=train_dicts,
    transform=data_transform,
    num_workers=8,
)
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
)

val_ds = CacheDataset(
    data=val_dicts,
    transform=data_transform,
    num_workers=8,
)
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
)

autoencoder = AutoencoderKL(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(32, 32, 64),
    latent_channels=3,
    num_res_blocks=1,
    norm_num_groups=16,
    attention_levels=(False, False, True),
).to(device)

discriminator = PatchDiscriminator(
    spatial_dims=3,
    num_layers_d=3,
    num_channels=32,
    in_channels=1,
    out_channels=1,
).to(device)

l1_loss = L1Loss()
adv_loss = PatchAdversarialLoss(criterion='least_squares')

loss_perceptual = PerceptualLoss(
    spatial_dims=3,
    network_type='squeeze',
    is_fake_3d=True,
    fake_3d_ratio=0.2,
).to(device)

optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=1e-4)
optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=1e-4)

scaler_g = GradScaler()
scaler_d = GradScaler()

wandb.login()
run = wandb.init(project='autoencoderkl', name='version-1')

for epoch in range(n_epochs):
    autoencoder.train()
    discriminator.train()
    epoch_loss = 0
    gen_epoch_loss = 0
    disc_epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader, start=1), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in progress_bar:
        images = random.choice([batch['mr'], batch['ct']]).to(device)
        optimizer_g.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            reconstruction, z_mu, z_sigma = autoencoder(images)
            kl_loss = KL_loss(z_mu, z_sigma)

            recons_loss = l1_loss(reconstruction.float(), images.float())
            p_loss = loss_perceptual(reconstruction.float(), images.float())
            loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

            if epoch > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g += adv_weight * generator_loss

        scaler_g.scale(loss_g).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()

        if epoch > autoencoder_warm_up_n_epochs:
            with autocast(enabled=True):
                optimizer_d.zero_grad(set_to_none=True)
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                loss_d = adv_weight * discriminator_loss

            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

        epoch_loss += recons_loss.item()
        if epoch > autoencoder_warm_up_n_epochs:
            gen_epoch_loss += generator_loss.item()
            disc_epoch_loss += discriminator_loss.item()

        progress_bar.set_postfix({
            "recons_loss": epoch_loss / (step + 1),
            "gen_loss": gen_epoch_loss / (step + 1),
            "disc_loss": disc_epoch_loss / (step + 1),
        })

    if (epoch + 1) % val_interval == 0:
        autoencoder.eval()
        val_loss = 0

        with torch.no_grad():
            for val_step, batch in tqdm(enumerate(val_loader, start=1), total=len(val_loader), ncols=110):
                images = random.choice([batch['mr'], batch['ct']]).to(device)

                with autocast(enabled=True):
                    reconstruction = autoencoder.reconstruct(images)
                    recons_loss = l1_loss(images.float(), reconstruction.float())

                val_loss += recons_loss.item()

            batch = next(iter(val_loader))
            images = random.choice([batch['mr'], batch['ct']]).to(device)

            with autocast(enabled=True):
                reconstruction = autoencoder.reconstruct(images)
                reconstruction = reconstruction.clip(0, 1)

            grid = make_grid(torch.cat([
                images[:4, :, :, :, 32].cpu(),
                reconstruction[:4, :, :, :, 32].cpu(),
            ], dim=0), nrow=4, value_range=(0, 1), normalize=True, scale_each=True)

        print(f"epoch {epoch + 1} val loss: {val_loss / val_step:.4f}")
        wandb.log({
            'epoch': epoch + 1,
            'recons_loss': epoch_loss / step,
            'gen_loss': gen_epoch_loss / step,
            'disc_loss': disc_epoch_loss / step,
            'val_loss': val_loss / val_step,
            'sample': wandb.Image(grid.permute(1, 2, 0), caption='Image, Recons'),
        })

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(autoencoder.state_dict(), 'best_model.pth')
