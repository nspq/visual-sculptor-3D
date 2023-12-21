# Data Processing code
# Usage:
# python3 02_feature_extraction.py ../data3 data
# note: path provided is relative to src directory


import cv2
import numpy as np
import os
import sys
from natsort import natsorted

def findHomo(image1, image2):

  # Convert images to grayscale
  gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
  gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

  # Initialize SIFT detector
  sift = cv2.SIFT_create()

  # Find the keypoints and descriptors with SIFT
  kp1, des1 = sift.detectAndCompute(gray1, None)
  kp2, des2 = sift.detectAndCompute(gray2, None)

  # Use the FLANN (Fast Library for Approximate Nearest Neighbors) Matcher for SIFT
  FLANN_INDEX_KDTREE = 1
  index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
  search_params = dict(checks=50)
  flann = cv2.FlannBasedMatcher(index_params,search_params)
  matches = flann.knnMatch(des1, des2, k=2)

  # Apply ratio test to find good matches
  good_matches = []
  for m, n in matches:
      if m.distance < 0.75 * n.distance:
          good_matches.append(m)

  # Extract matched keypoints
  src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
  dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

  # Calculate the homography matrices M1 and M2
  M1, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

  # Reverse the order to calculate the inverse transformation
  M2, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

  return M1, M2

alpha_values = [0.5]
interpolated_results = []
alpha = 0.5

# Recursively create warped images from angles in between original two
def recurse_interp(im1, im2, c):
  if c == 0:
    return 0
  elif c >= 0:

    m1, m2 = findHomo(im1, im2)

    i_matrix = alpha * m1 + (1 - alpha) * m1
    h, w = im1.shape[:2]
    i_result = cv2.warpPerspective(im1, i_matrix, (w, h))
    #interpolated_results.append(interpolated_result)
    cv2_imshow(i_result)


    recurse_interp(im2, im1, c - 1)
    #recurse_interp(i_result, im2, c - 1)



# testWarp = recurse_interp(building1, building2, 2)




def crop_images(initial_path, final_path):
    # Check if the final image path exists, create if not
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    # Get a list of all image files in the initial path and sort them
    image_files = natsorted([f for f in os.listdir(initial_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    # Iterate over each image file
    for idx, image_file in enumerate(image_files):
        print(image_file)
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


