import torch
import numpy as np
from torchvision.utils import make_grid, save_image
import os
from PIL import Image
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt


image_path = 'real_test/original_image_0.png'
image = Image.open(image_path)
image = F.to_tensor(image).unsqueeze(0)

# norm_image = image / 255.0
# sqrt_images = torch.sqrt(image)
# sigma_start_noise =  0.1  #1 / np.sqrt(255)
# gaussian_noise = torch.randn(sqrt_images.size()) * sigma_start_noise
# noisy_init_samples = sqrt_images + gaussian_noise
# noisy_init_samples = noisy_init_samples * sqrt_images
# noisy_init_samples = torch.clamp(noisy_init_samples, min=0, max=1)

# image_array = np.array(image)
# noisy_image = np.random.poisson(image)

print(torch.max(image))
print(torch.min(image))

# plt.hist(noisy_init_samples.view(-1).cpu().numpy(), bins=100)
# plt.savefig('test/hist.png')


# save_image(noisy_init_samples ,'test/test_image_0_255.png')
