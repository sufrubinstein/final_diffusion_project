import os
from PIL import Image

# Folder containing the images
image_folder = '/home-sipl/prj7565/ncsnv2/exp/datasets/celeba_test/celeba/img_align_celeba'

# List to store invalid image files
corrupted_images = []

# Iterate through all files in the folder
for filename in os.listdir(image_folder):
    file_path = os.path.join(image_folder, filename)

    try:
        # Try to open the image
        with Image.open(file_path) as img:
            img.verify()  # Check if it's a valid image

    except (IOError, SyntaxError) as e:
        # If an error occurs, the image is corrupted or invalid
        print(f"Corrupted image found: {filename}")
        corrupted_images.append(filename)

# Output the corrupted images
if corrupted_images:
    print("\nCorrupted or invalid images:")
    for img in corrupted_images:
        print(img)
else:
    print("All images are valid.")