import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import open3d as o3d
from skimage import img_as_ubyte
from skimage.io import imsave
from skimage.color import gray2rgb
import os

# Feature extraction functions
def estimate_relative_poses(matches, keypoints_list):
    poses = []  # List to store relative camera poses

    # Iterate through pairs of consecutive images
    for i in range(len(matches) - 1):
        # Extract keypoints and matches for the current pair
        keypoints1 = keypoints_list[i]
        keypoints2 = keypoints_list[i + 1]
        matches_i = matches[i]

        # Extract corresponding points from keypoints
        pts1 = np.float32([keypoints1[m['queryIdx']][:2] for m in matches_i]).reshape(-1, 1, 2)
        pts2 = np.float32([keypoints2[m['trainIdx']][:2] for m in matches_i]).reshape(-1, 1, 2)

        # Estimate fundamental matrix
        fundamental_matrix, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

        # Decompose essential matrix into rotation and translation
        essential_matrix, _ = cv2.findEssentialMat(pts1, pts2)
        _, R, t, _ = cv2.recoverPose(essential_matrix, pts1, pts2)

        # Create a 4x4 transformation matrix
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t.flatten()

        # Accumulate poses
        if i == 0:
            poses.append(np.eye(4))  # First pose is identity
        poses.append(np.dot(poses[-1], pose))

    return poses

def triangulate_3d_points(matches, keypoints_list, poses):
    triangulated_points = []  # List to store triangulated 3D points

    # Iterate through pairs of consecutive images
    for i in range(len(matches) - 1):
        # Extract keypoints and matches for the current pair
        keypoints1 = keypoints_list[i]
        keypoints2 = keypoints_list[i + 1]
        matches_i = matches[i]

        # Extract corresponding points from keypoints
        pts1 = np.float32([keypoints1[m['queryIdx']][:2] for m in matches_i]).reshape(-1, 1, 2)
        pts2 = np.float32([keypoints2[m['trainIdx']][:2] for m in matches_i]).reshape(-1, 1, 2)

        # Projection matrices for the current and next images
        P1 = np.eye(4)
        P2 = poses[i + 1]

        # Triangulate 3D points
        points_4d_homogeneous = cv2.triangulatePoints(P1[:3], P2[:3], pts1, pts2)

        # Convert homogeneous coordinates to 3D
        points_3d = points_4d_homogeneous[:3] / points_4d_homogeneous[3]

        # Append the triangulated points to the list
        triangulated_points.append(points_3d.T)

    # Stack the triangulated points into a single array
    triangulated_points = np.vstack(triangulated_points)

    return triangulated_points

def bundle_adjustment(poses, points, keypoints_list):
    # Implement code for bundle adjustment here (optional)
    # ...
    return

def visualize_3d_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, and z coordinates from the 3D points
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Point Cloud Visualization')

    plt.show()

# 3D Reconstruction functions
def create_point_cloud(triangulated_points):
    # Use a library like open3d to create a point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(triangulated_points)
    return point_cloud

def create_mesh(point_cloud, faces):
    # Use a library like Trimesh to create a mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = point_cloud.points
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh

def generate_3d_model(mesh, texture_images):
    # Generate a 3D model by mapping textures onto the mesh
    # ...
    return

def visualize_3d_data(point_cloud, mesh):
    # Visualize the point cloud and mesh using open3d or other libraries
    o3d.visualization.draw_geometries([point_cloud, mesh])
    return

def main():
    # Set the working directory to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # Load matches and keypoints
    matches_file_paths = ['data/matches_image0_to_image1.npy', 'data/matches_image1_to_image2.npy', 'data/matches_image2_to_image3.npy']
    matches = [np.load(match_file_path, allow_pickle=True) for match_file_path in matches_file_paths]

    keypoints_file_paths = ['data/keypoints_image0.npy', 'data/keypoints_image1.npy', 'data/keypoints_image2.npy', 'data/keypoints_image3.npy']
    keypoints_list = [np.load(kp_file_path) for kp_file_path in keypoints_file_paths]

    # Estimate relative camera poses
    poses = estimate_relative_poses(matches, keypoints_list)

    # Triangulate 3D points
    triangulated_points = triangulate_3d_points(matches, keypoints_list, poses)

    # Bundle Adjustment (optional)
    # poses, triangulated_points = bundle_adjustment(poses, triangulated_points, keypoints_list)

    # Visualize 3D points
    visualize_3d_points(triangulated_points)

    # Create Point Cloud
    point_cloud = create_point_cloud(triangulated_points)

    # Create Mesh
    faces = []  # Define or obtain faces based on your requirements
    mesh = create_mesh(point_cloud, faces)

    # Generate 3D Model (Optional: Texture Mapping)
    # 3D_model = generate_3d_model(mesh, texture_images)

    # Visualize 3D Data
    visualize_3d_data(point_cloud, mesh)

if __name__ == "__main__":
    main()

