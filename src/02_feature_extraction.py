# Feature extraction code
# Implement feature extraction here
# Import necessary libraries (e.g., OpenCV, scikit-image)
import cv2
import numpy as np
import os


# Set the working directory to the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

# List of image paths
image_paths = ['data/images/image0.jpg', 'data/images/image1.jpg', 'data/images/image2.jpg', 'data/images/image3.jpg']

# Initialize the SIFT feature extractor
sift = cv2.SIFT_create()

keypoints_list = []  # List to store keypoints for each image
descriptors_list = []  # List to store descriptors for each image

# Iterate through the images and perform feature extraction
for image_path in image_paths:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Get the size of the loaded image
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_size = (gray_image.shape[1], gray_image.shape[0])  # (width, height)

    # Create a black image with the same size as canvas
    blank_image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    # Draw keypoints on the black image
    image_with_keypoints = cv2.drawKeypoints(blank_image, keypoints, outImage=None)
    
    # Show the image with keypoints
    fname = f"outputs/imgOutputVis{image_path[17]}.jpg"

    cv2.imwrite(fname, image_with_keypoints)
    #cv2.imshow("Keypoints of Image", image_with_keypoints)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    # Convert keypoints to a list of (x, y, size, angle, response, octave, class_id)
    keypoints_data = [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]
    
    keypoints_list.append(keypoints_data)
    descriptors_list.append(descriptors)

# Save the keypoints as a list of lists
for i in range(len(image_paths)):
    np.save(f'data/keypoints_image{i}.npy', keypoints_list[i])
    np.save(f'data/descriptors_image{i}.npy', descriptors_list[i])

