import cv2
import numpy as np
import os

def match_features(descriptors1, descriptors2, bf, ratio_threshold=0.75):
    # Perform brute-force matching
    matches_i = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to get good matches
    good_matches = [m for m, n in matches_i if m.distance < ratio_threshold * n.distance]

    return good_matches

def display_matches(img1, img2, keypoints1, keypoints2, matches, i1, i2, outputs_dir):
    # Create a new image with double the width and height of the larger image
    height = max(img1.shape[0], img2.shape[0]) * 2
    width = max(img1.shape[1], img2.shape[1]) * 2
    new_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Place the two images side by side on the new image
    new_image[:img1.shape[0], :img1.shape[1]] = img1
    new_image[:img2.shape[0], img1.shape[1]:] = img2

    # Draw lines connecting the matches
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        # Coordinates in the larger combined image
        pt1 = (int(keypoints1[img1_idx][0]), int(keypoints1[img1_idx][1]))
        pt2 = (int(keypoints2[img2_idx][0] + img1.shape[1]), int(keypoints2[img2_idx][1]))

        cv2.line(new_image, pt1, pt2, (0, 255, 0), 8)

    # Save the resulting image
    fname = os.path.join(outputs_dir, f"imageMatch{i1}and{i2}.jpg")
    cv2.imwrite(fname, new_image)

def main():
    # Set the working directory to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    data_dir = os.path.join(project_root, 'src', 'data')

    # Dynamically get image paths
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_paths = [os.path.join(data_dir, f) for f in image_files]
    
    keypoints_dir = os.path.join(data_dir, "keypoints")
    descriptors_dir = os.path.join(data_dir, "descriptors")
    matches_dir = os.path.join(data_dir, "matches")
    outputs_dir = os.path.join(data_dir, "outputs")

    # Ensure that the matches directory exists or create it
    os.makedirs(matches_dir, exist_ok=True)

    # Dynamically get keypoints and descriptors file paths
    keypoints_file_paths = [os.path.join(keypoints_dir, f) for f in os.listdir(keypoints_dir) if f.startswith('keypoints')]
    descriptors_file_paths = [os.path.join(descriptors_dir, f) for f in os.listdir(descriptors_dir) if f.startswith('descriptors')]

    # Sort the file paths for consistency
    image_paths.sort()
    keypoints_file_paths.sort()
    descriptors_file_paths.sort()

    # Initialize the feature matcher (in this case, Brute-Force matcher)
    bf = cv2.BFMatcher()

    # Match descriptors between pairs of images
    matches_list = []  # List to store matches for each pair of images

    # Load keypoints for each image
    keypoints_list = [np.load(kp_file_path) for kp_file_path in keypoints_file_paths]

    for i in range(len(keypoints_file_paths) - 1):
        descriptors1 = np.load(descriptors_file_paths[i])
        descriptors2 = np.load(descriptors_file_paths[i + 1])

        # Perform feature matching
        good_matches = match_features(descriptors1, descriptors2, bf)

        # Implement RANSAC to estimate homography
        src_pts = np.float32([keypoints_list[i][m.queryIdx][:2] for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_list[i + 1][m.trainIdx][:2] for m in good_matches]).reshape(-1, 1, 2)

        # Use RANSAC to estimate homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Apply the estimated homography to the first image
        img1 = cv2.imread(image_paths[i])
        warped_img1 = cv2.warpPerspective(img1, M, (img1.shape[1] + img1.shape[1], img1.shape[0]))

        # Resize the warped image to match the size of the second image
        warped_img1 = cv2.resize(warped_img1, (cv2.imread(image_paths[i + 1]).shape[1], cv2.imread(image_paths[i + 1]).shape[0]))

        # Concatenate the warped image and the second image
        result = np.concatenate([warped_img1, cv2.imread(image_paths[i + 1])], axis=1)

        # Save the resulting image
        fname = os.path.join(outputs_dir, f"imageMatchRANSAC{i}and{i + 1}.jpg")
        cv2.imwrite(fname, result)

        # Convert cv2.DMatch objects to dictionaries
        matches_to_save = [{'queryIdx': m.queryIdx, 'trainIdx': m.trainIdx, 'distance': m.distance} for m in good_matches]

        # Save the RANSAC matches as an .npy file
        np.save(os.path.join(matches_dir, f'matches_image{i}_to_image{i + 1}.npy'), matches_to_save)

        # Call display_matches function to draw and save matches
        display_matches(img1, cv2.imread(image_paths[i + 1]), keypoints_list[i], keypoints_list[i + 1], good_matches, i, i + 1, outputs_dir)



if __name__ == "__main__":
    main()
