# fix_image_orientation.py

from PIL import Image
import os

# Path to your image
input_path = "assets/profile_picture.jpg"
output_path = "assets/profile_picture_fixed.jpg"

# Open image and remove EXIF rotation manually
image = Image.open(input_path)
if hasattr(image, '_getexif'):  # Only works for JPEGs with EXIF data
    exif = image._getexif()
    if exif is not None:
        orientation = exif.get(274, 1)  # 274 is the orientation tag
        if orientation == 3:
            image = image.rotate(180, expand=True)
        elif orientation == 6:
            image = image.rotate(-90, expand=True)  # Rotate 90 degrees clockwise
        elif orientation == 8:
            image = image.rotate(90, expand=True)   # Rotate 90 degrees counterclockwise

# Save the fixed image
image.save(output_path, quality=95, optimize=True)
print("✅ Image orientation fixed and saved as:", output_path)