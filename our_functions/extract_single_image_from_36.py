from PIL import Image

# Load the PNG file
image = Image.open("new_noise_adding_method/image_for_sigma126.png")

# Define the size of each image in the grid
image_width, image_height = 128, 128

# Choose the position of the image you want to extract
# For example, to get the image at row 0, column 0 (top-left corner)
row, col = 0, 0

# Calculate the coordinates for cropping
left = col * image_width
upper = row * image_height
right = left + image_width
lower = upper + image_height

# Crop the single 128x128 image
single_image = image.crop((left, upper, right, lower))

# Save or display the cropped image
single_image.save("images_for_psnr/extracted_image.png")
single_image.show()

from PIL import Image

# Load the image
image = Image.open("images_for_psnr/extracted_image.png")

# Convert the image to grayscale to detect black borders more easily
gray_image = image.convert("L")

# Get bounding box of the non-black content
bbox = gray_image.getbbox()

# Crop the image to this bounding box
cropped_image = image.crop(bbox)

# Save or display the cropped image
cropped_image.save("images_for_psnr/extracted_image.png")
cropped_image.show()

print(cropped_image.size)

# Resize the image to 128x128
resized_image = cropped_image.resize((128, 128),  Image.LANCZOS)
resized_image.save("images_for_psnr/extracted_image.png")
print(resized_image.size)