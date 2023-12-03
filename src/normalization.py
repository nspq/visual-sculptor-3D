import cv2
import os
import matplotlib.pyplot as plt

def normalize_image(image_path, output_path, target_size=(256, 256)):
    # Read the image
    img = cv2.imread(image_path)

    # Resize the image to the target size
    img_resized = cv2.resize(img, target_size)

    # Normalize pixel values to the range [0, 1]
    img_normalized = img_resized / 255.0

    # Save the normalized image
    cv2.imwrite(output_path, img_normalized * 255.0)

    return img_normalized

def sift_features(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the SIFT algorithm from the opencv-contrib-python package
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Draw keypoints on the image
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

    return img_with_keypoints

def display_images(images, titles):
    num_images = len(images)
    num_rows = 2
    num_cols = (num_images + 1) // num_rows

    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(images[i], cmap='gray' if len(images[i].shape) == 2 else None)
        plt.title(titles[i])
        plt.axis('off')

    plt.show()

if __name__ == "__main__":
    # Provide the paths to your six photos
    photo_paths = [
        "p1.jpg",
        "p2.jpg",
        "p3.jpg",
        "p4.jpg",
        "p5.jpg",
        "p6.jpg",
        "p7.jpg",
        "p8.jpg",
        "p9.jpg",
        "p10.jpg",
        "p11.jpg",
        "p12.jpg",
        "p13.jpg",
        "p14.jpg",
        "p15.jpg",
        "p16.jpg",
        "p17.jpg",
        "p18.jpg",
        "p19.jpg",
        "p20.jpg",

    ]

    # Output directory for normalized photos
    output_directory = "normalized_photos"

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    normalized_images = []
    sift_images = []
    for photo_path in photo_paths:
        photo_name = os.path.basename(photo_path)
        output_path = os.path.join(output_directory, f"normalized_{photo_name}")
        normalized_image = normalize_image(photo_path, output_path)
        normalized_images.append(normalized_image)

        # Apply SIFT to the normalized image
        sift_image = sift_features((normalized_image * 255).astype('uint8'))
        sift_images.append(sift_image)

    # Display the original, normalized, and SIFT images side by side
    original_images = [cv2.imread(photo_path) for photo_path in photo_paths]
    display_images(original_images + normalized_images + sift_images,
                   ['Original']*len(original_images)+ ['Normalized']*len(normalized_images) + ['SIFT']*len(sift_images))