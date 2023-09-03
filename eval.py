import os
import glob
import numpy as np

from tqdm import tqdm
from metric import ImageMetrics

test_dir = 'dataset/train/Task1/val'
pred_dir = 'dataset/train/Task1/pred'

true_ct_images = sorted(glob.glob(os.path.join(test_dir, '*', 'ct.nii.gz')))
true_mask_images = sorted(glob.glob(os.path.join(test_dir, '*', 'mask.nii.gz')))
pred_ct_images = sorted(glob.glob(os.path.join(pred_dir, '*', 'ct.nii.gz')))

metrics = ImageMetrics()
stats = []

for true_image, true_mask, pred_image in tqdm(zip(true_ct_images, true_mask_images, pred_ct_images)):
    scores = metrics.score_patient(true_image, pred_image, true_mask)
    stats.append(scores)

print('mean mae: ', np.mean([score['mae'] for score in stats]))
print('mean ssim: ', np.mean([score['ssim'] for score in stats]))
print('mean psnr: ', np.mean([score['psnr'] for score in stats]))

print('std mae: ', np.std([score['mae'] for score in stats]))
print('std ssim: ', np.std([score['ssim'] for score in stats]))
print('std psnr: ', np.std([score['psnr'] for score in stats]))
