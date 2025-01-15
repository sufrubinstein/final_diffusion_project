import os
import numpy as np
from PIL import Image
import math
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms import functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import matplotlib.pyplot as plt


# Function to calculate PSNR
def calculate_psnr(original_img, denoised_img):
    # Convert images to numpy arrays
    original = np.array(original_img)
    denoised = np.array(denoised_img)

    # Calculate Mean Squared Error (MSE)
    mse = np.mean((original - denoised) ** 2)

    # If MSE is zero, PSNR is infinite (images are identical)
    if mse == 0:
        return float('inf')

    # Maximum pixel value of the image
    max_pixel = 255.0

    # Calculate PSNR
    psnr = 10 * math.log10((max_pixel ** 2) / mse)
    return psnr

# Create a folder for saving images if it doesn't exist
# output_folder = '/home-sipl/prj7565/ncsnv2/images_for_psnr'
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# Save the images in the "picture_psnr" folder
# original_img_rgb.save(os.path.join(output_folder, 'original_resized_rgb.png'))
# denoised_img_rgb.save(os.path.join(output_folder, 'denoised_rgb.png'))

#Function for FID
def calculate_fid(original_img, denoised_img):

    original = np.array(original_img)
    denoised = np.array(denoised_img)
    
    # Reshape to 2D (features as rows)
    original = original.reshape(-1, 3)  # Each row is a pixel with RGB values
    denoised = denoised.reshape(-1, 3)
    
    # Calculate mean and covariance
    mu1, sigma1 = np.mean(original, axis=0), np.cov(original, rowvar=False)
    mu2, sigma2 = np.mean(denoised, axis=0), np.cov(denoised, rowvar=False)

    # print(np.shape(mu1))
    # print(np.shape(sigma1))
    
    # Compute mean difference squared
    diff = np.sum((mu1 - mu2) ** 2)
    
    # Compute covariance mean (sqrt of product of covariances)
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    
    # Handle imaginary values from sqrtm
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Compute FID score
    fid = diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    fid = np.sqrt(fid)

    # print(np.max(denoised))
    # print(np.min(denoised))

    return fid

# sigma_check = np.linspace(1, 0.01, 40)

# mean_psnr_arr = []
# mean_fid_arr = []
# mean_ssim_arr = []

# noise_sigma = '025'

# for sigma in sigma_check:
psnr_arr = []
fid_arr = []
ssim_arr = []

for i in range(8):

    # Load the images before and after denoising
    original_img = Image.open("gaus_702/original_image_{}.png".format(i))
    denoised_img = Image.open("0625_train_1/denoised_image_{}.png".format(i))
    # denoised_img = Image.open("test_sigma_{}/test_sigma_{}_{}.png".format(noise_sigma , sigma , i))

    # Resize the original image to 128x128 to match the denoised image
    original_img_resized = original_img.resize((128, 128), Image.LANCZOS)

    # Convert both images to RGB to ensure they have 3 channels
    original_img_rgb = original_img_resized.convert('RGB')
    denoised_img_rgb = denoised_img.convert('RGB')

    original_img_rgb_tensor = F.to_tensor(original_img_rgb).unsqueeze(0)  # Shape: (1, 3, H, W)
    denoised_img_rgb_tensor = F.to_tensor(denoised_img_rgb).unsqueeze(0)

    # print("tensor")
    # print(torch.max(original_img_rgb_tensor))
    # print(torch.min(original_img_rgb_tensor))
    # print(torch.max(denoised_img_rgb_tensor))
    # print(torch.min(denoised_img_rgb_tensor))

    original_img_rgb_array = np.array(original_img_rgb)
    denoised_img_rgb_array = np.array(denoised_img_rgb)

    # print(f"original_img_rgb_array: {original_img_rgb_array.shape}")
    # print(f"denoised_img_rgb_array: {denoised_img_rgb_array.shape}")

    # print("array")
    # print(np.max(original_img_rgb_array))
    # print(np.min(original_img_rgb_array))
    # print(np.max(denoised_img_rgb_array))
    # print(np.min(denoised_img_rgb_array))

    # original_img_rgb_norm = original_img_rgb_tensor / 255.0
    # denoised_img_rgb_norm = denoised_img_rgb_tensor / 255.0

    # Calculate PSNR
    psnr_value = psnr(original_img_rgb_array, denoised_img_rgb_array)
    fid_value = calculate_fid(original_img_rgb_array , denoised_img_rgb_array)
    ssim_value = ssim(original_img_rgb_array, denoised_img_rgb_array , channel_axis=2)
    
    psnr_arr.append(psnr_value)
    fid_arr.append(fid_value)
    ssim_arr.append(ssim_value)


#print PSNR
print(f"PSNR values are: {np.round(psnr_arr,2)} dB")
print(f"PSNR mean is: {np.mean(psnr_arr):.2f} dB")
print(f"PSNR std is: {np.std(psnr_arr):.2f} dB")

#print FID
print(f"FID values are: {np.round(fid_arr,2)}")
print(f"FID mean is: {np.mean(fid_arr):.2f}")
print(f"FID std is: {np.std(fid_arr):.2f}")

#print SSIM
print(f"SSIM values are: {np.round(ssim_arr,2)}")
print(f"SSIM mean is: {np.mean(ssim_arr):.2f}")
print(f"SSIM std is: {np.std(ssim_arr):.2f}")

#     mean_psnr_arr.append(np.mean(psnr_arr))
#     mean_fid_arr.append(np.mean(fid_arr))
#     mean_ssim_arr.append(np.mean(ssim_arr))


# plt.figure()
# plt.plot(sigma_check, mean_psnr_arr, label='PSNR')
# plt.xlabel('Test Sigma')
# plt.ylabel('PSNR')
# plt.title('PSNR vs Test Sigma (Noise sigma = 0.{})'.format(noise_sigma))
# plt.tight_layout()
# plt.grid()
# plt.savefig('test_sigma_graphs/0.{}_psnr_vs_sigma.png'.format(noise_sigma))

# plt.figure()
# plt.plot(sigma_check, mean_fid_arr, label='PSNR')
# plt.xlabel('Test Sigma')
# plt.ylabel('FID')
# plt.title('FID vs Test Sigma (Noise sigma = 0.{})'.format(noise_sigma))
# plt.tight_layout()
# plt.grid()
# plt.savefig('test_sigma_graphs/0.{}_fid_vs_sigma.png'.format(noise_sigma))

# plt.figure()
# plt.plot(sigma_check, mean_ssim_arr, label='PSNR')
# plt.xlabel('Test Sigma')
# plt.ylabel('SSIM')
# plt.title('SSIM vs Test Sigma (Noise sigma = 0.{})'.format(noise_sigma))
# plt.tight_layout()
# plt.grid()
# plt.savefig('test_sigma_graphs/0.{}_ssim_vs_sigma.png'.format(noise_sigma))

