import cv2
import numpy as np
import os

def match_features(descriptors1, descriptors2, bf, ratio_threshold=0.75):
    # Perform brute-force matching
    matches_i = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to get good matches
    good_matches = [m for m, n in matches_i if m.distance < ratio_threshold * n.distance]

    return good_matches

def display_matches(img1, img2, keypoints1, keypoints2, matches, i1, i2):
    # Create a new image with double the width and height of the larger image
    height = max(img1.shape[0], img2.shape[0]) * 2
    width = max(img1.shape[1], img2.shape[1]) * 2
    new_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Place the two images side by side on the new image
    new_image[:img1.shape[0], :img1.shape[1]] = img1
    new_image[:img2.shape[0], img1.shape[1]:] = img2

    # Draw lines connecting the matches
    # Draw lines connecting the matches
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        # Coordinates in the larger combined image
        pt1 = (int(keypoints1[img1_idx][0]), int(keypoints1[img1_idx][1]))
        pt2 = (int(keypoints2[img2_idx][0] + img1.shape[1]), int(keypoints2[img2_idx][1]))

        cv2.line(new_image, pt1, pt2, (0, 255, 0), 8)


    # Save the resulting image
    cv2.imwrite(f'outputs/imageMatch{i1}and{i2}.jpg', new_image)

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

        # Display and save the matches
        img1 = cv2.imread(image_paths[i])
        img2 = cv2.imread(image_paths[i + 1])
        display_matches(img1, img2, keypoints_list[i], keypoints_list[i + 1], good_matches, i, i+1)

    # Save the matches for each pair of images (optional)
    for i, match_list in enumerate(matches):
        matches_to_save = [{'queryIdx': m.queryIdx, 'trainIdx': m.trainIdx, 'distance': m.distance} for m in match_list]
        np.save(f'data/matches_image{i}_to_image{i + 1}.npy', matches_to_save)

if __name__ == "__main__":
    main()
