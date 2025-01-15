import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import torchvision.transforms.functional as TF  # To convert tensor back to PIL image

# Path to the saved image
saved_image_path = '/home-sipl/prj7565/ncsnv2/suf_testing/171053.jpg'

# Output folder for saving the transformed image
output_folder = '/home-sipl/prj7565/ncsnv2/suf_testing'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the transformation
tran_transform = test_transform = transforms.Compose([
    transforms.Resize((178, 178)),  # Ensure this matches your image size
    transforms.ToTensor()  # Convert to tensor
])

# Load, apply transformation, and save the transformed image
try:
    with Image.open(saved_image_path) as img:
        # Apply the transformation
        transformed_img_tensor = tran_transform(img)
        
        # Convert the transformed tensor back to a PIL image
        transformed_img = TF.to_pil_image(transformed_img_tensor)
        
        # Save the transformed image
        transformed_img.save(os.path.join(output_folder, 'transformed_171053_178.jpg'))
        print(f"Transformed image saved at: {os.path.join(output_folder, 'transformed_171053.jpg')}")
        
except Exception as e:
    print(f"Error occurred while transforming and saving the image: {e}")
