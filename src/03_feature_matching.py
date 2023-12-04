import cv2
import numpy as np
import os

def match_features(descriptors1, descriptors2, bf, ratio_threshold=0.75):
    # Perform brute-force matching
    matches_i = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to get good matches
    good_matches = [m for m, n in matches_i if m.distance < ratio_threshold * n.distance]

    return good_matches

def main():
    # Set the working directory to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    image_paths = ['data/images/image0.jpg', 'data/images/image1.jpg', 'data/images/image2.jpg', 'data/images/image3.jpg']

    # Load keypoints and descriptors for the images
    keypoints_file_paths = ['data/keypoints_image0.npy', 'data/keypoints_image1.npy', 'data/keypoints_image2.npy', 'data/keypoints_image3.npy']
    descriptors_file_paths = ['data/descriptors_image0.npy', 'data/descriptors_image1.npy', 'data/descriptors_image2.npy', 'data/descriptors_image3.npy']

    # Initialize the feature matcher (in this case, Brute-Force matcher)
    bf = cv2.BFMatcher()

    # Match descriptors between pairs of images
    matches = []  # List to store matches for each pair of images

    # Load keypoints for each image
    keypoints_list = [np.load(kp_file_path) for kp_file_path in keypoints_file_paths]

    for i in range(len(keypoints_file_paths) - 1):
        descriptors1 = np.load(descriptors_file_paths[i])
        descriptors2 = np.load(descriptors_file_paths[i + 1])

        # Perform feature matching
        good_matches = match_features(descriptors1, descriptors2, bf)

        matches.append(good_matches)

    # Save the matches for each pair of images (optional)
    for i, match_list in enumerate(matches):
        matches_to_save = [{'queryIdx': m.queryIdx, 'trainIdx': m.trainIdx, 'distance': m.distance} for m in match_list]
        np.save(f'data/matches_image{i}_to_image{i + 1}.npy', matches_to_save)

if __name__ == "__main__":
    main()
