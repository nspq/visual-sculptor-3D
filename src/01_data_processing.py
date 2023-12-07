# Data Processing code
# Usage:
# python3 02_feature_extraction.py ../data3 data
# note: path provided is relative to src directory


import cv2
import numpy as np
import os
import sys

def crop_images(initial_path, final_path):
    # Check if the final image path exists, create if not
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    # Get a list of all image files in the initial path and sort them
    image_files = sorted([f for f in os.listdir(initial_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    # Iterate over each image file
    for idx, image_file in enumerate(image_files):
        # Read the image
        img = cv2.imread(os.path.join(initial_path, image_file))

        # Get the center coordinates of the image
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2

        # Calculate the new crop dimensions (75% of the actual width and height)
        crop_width = int(0.75 * img.shape[1])
        crop_height = int(0.75 * img.shape[0])

        # Crop the image around the center to the specified dimensions
        cropped_img = img[center_y - crop_height // 2:center_y + crop_height // 2,
                          center_x - crop_width // 2:center_x + crop_width // 2]

        # Rename and save the processed image to the final path
        new_image_name = f"image{idx}.png"
        cv2.imwrite(os.path.join(final_path, new_image_name), cropped_img)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 02_feature_extraction.py initial/images/path final/images/path")
    else:
        i_images_path = sys.argv[1]
        f_images_path = sys.argv[2]

        crop_images(i_images_path, f_images_path)


