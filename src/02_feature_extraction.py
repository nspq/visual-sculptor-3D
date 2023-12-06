# Usage:
# python3 02_feature_extraction.py src/data


# Feature extraction code
# Implement feature extraction here
# Import necessary libraries (e.g., OpenCV, scikit-image)
import cv2
import numpy as np
import os
import sys


def resize_images(input_folder, output_folder, target_height, target_width):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            img = cv2.imread(image_path)

            # Resize the image
            resized_img = cv2.resize(img, (target_width, target_height))

            # Save the resized image, overwriting the original
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, resized_img)

def extract_features(image_dir):
    # Set the working directory to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Create output directories if they don't exist
    keypoints_dir = os.path.join(image_dir, 'keypoints')
    descriptors_dir = os.path.join(image_dir, 'descriptors')
    outputs_dir = os.path.join(image_dir, 'outputs')

    os.makedirs(keypoints_dir, exist_ok=True)
    os.makedirs(descriptors_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)


    # List image files in the specified directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # os.makedirs(os.path.join(image_dir, 'resized_images'), exist_ok=True)
    # resize_images(image_dir, os.path.join(image_dir, 'resized_images'), 600, 1300)
    image_paths = [os.path.join(image_dir, f) for f in image_files]

    # Initialize the SIFT feature extractor
    sift = cv2.SIFT_create()

    keypoints_list = []  # List to store keypoints for each image
    descriptors_list = []  # List to store descriptors for each image

    # Iterate through the images and perform feature extraction
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(image, None)

        # Draw keypoints on the black image
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, outImage=None)

        # Show the image with keypoints
        fname = os.path.join(outputs_dir, f"imgOutputVis{os.path.splitext(os.path.basename(image_path))[0]}.jpg")

        cv2.imwrite(fname, image_with_keypoints)

        # Convert keypoints to a list of (x, y, size, angle, response, octave, class_id)
        keypoints_data = [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

        keypoints_list.append(keypoints_data)
        descriptors_list.append(descriptors)

    # Save the keypoints as a list of lists
    for i in range(len(image_files)):
        np.save(os.path.join(keypoints_dir, f'keypoints_image{i}.npy'), keypoints_list[i])
        np.save(os.path.join(descriptors_dir, f'descriptors_image{i}.npy'), descriptors_list[i])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 02_feature_extraction.py /path/to/image/directory")
    else:
        image_directory = sys.argv[1]
        extract_features(image_directory)
