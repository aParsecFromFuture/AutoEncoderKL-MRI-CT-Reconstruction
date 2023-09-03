import os
import glob
from tqdm import tqdm

import torch

import monai
from monai import transforms
from monai.inferers import SlidingWindowInferer
from monai.data import NibabelWriter
from generative.networks.nets import AutoencoderKL

data_transform = transforms.Compose([
    transforms.LoadImaged(keys=['mr'], image_only=True),
    transforms.EnsureChannelFirstd(keys=['mr']),
    transforms.EnsureTyped(keys=['mr']),
    transforms.Orientationd(keys=['mr'], axcodes='RAS'),
    transforms.ScaleIntensityd(keys=['mr'], minv=0, maxv=1),
])

model_path = 'best_model.pth'
mr_dir = 'dataset/train/Task1/test'
pred_dir = 'dataset/train/Task1/pred'

mr_images = glob.glob(os.path.join(mr_dir, '*', 'mr.nii.gz'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoencoderKL(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(32, 32, 64),
    latent_channels=3,
    num_res_blocks=1,
    norm_num_groups=16,
    attention_levels=(False, False, True),
).to(device)

model.load_state_dict(torch.load('best_model.pth'))
model.eval()

inferer = SlidingWindowInferer(
    roi_size=(64, 64, 64),
    sw_batch_size=4,
    overlap=0.25,
    sw_device=device,
    device='cpu',
)

writer = NibabelWriter()

for mr_image in tqdm(mr_images):
    data = data_transform({'mr': mr_image})
    image = data['mr'].unsqueeze(0)

    with torch.no_grad():
        output = inferer(image, model.reconstruct)

    output = output.squeeze(0)

    splits = mr_image.split('/')
    mr_image = splits[-1]
    patient_id = splits[-2]

    writer.set_data_array(output)
    writer.set_metadata(image.meta)

    os.makedirs(os.path.join(pred_dir, patient_id), exist_ok=True)
    writer.write(os.path.join(pred_dir, patient_id, mr_image))

