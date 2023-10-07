import open3d as o3d
import numpy as np
import math as m
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.models import build_model
import torch

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os 
import json

def create_mesh_cylinder_detection_based_on_alpha(R, t, alpha, radius=0.5, height=10):
    """
    Create a cylinder mesh based on given alpha (transparency) value.
    
    Parameters:
    - R: Rotation matrix
    - t: Translation vector
    - alpha: Transparency value (expected to be normalized between 0 and 1, with 1 being opaque)
    - radius: Radius of the cylinder
    - height: Height of the cylinder
    
    Returns:
    - cylinder: Cylinder mesh with color and transparency based on alpha
    """
    
    # Create cylinder
    cylinder = o3d.geometry.TriangleMesh().create_cylinder(radius, height)
    vertices = np.asarray(cylinder.vertices)[:, [2, 1, 0]]
    vertices[:, 0] += height / 2
    vertices = np.dot(R, vertices.T).T + t
    cylinder.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    
    green = np.array([0, 1, 0])
    white = np.array([1, 0, 0])
    
    color = alpha * green + (1 - alpha) * white
    colors = np.expand_dims(color, axis=0)
    colors = np.repeat(colors, vertices.shape[0], axis=0)
    cylinder.vertex_colors = o3d.utility.Vector3dVector(colors)

    return cylinder

def check_collision_with_suction_gripper_using_voxelgrid(voxel_grid, suction_location, suction_direction, voxel_size=1, grasp_length=10.0):
    """
    Check collision using VoxelGrid.
    
    Parameters:
    - point_cloud: Open3D point cloud object
    - suction_location: Starting point of the ray (suction tip)
    - suction_direction: Direction of the ray (suction normal)
    - voxel_size: Size of each voxel
    - grasp_length: Length of the ray (suction grasp depth)

    Returns:
    - True if collision detected, False otherwise
    """
    
    # Create a voxel grid from the point cloud
    #o3d.visualization.draw_geometries([voxel_grid])

    start_distance = 2.0
    end_distance = grasp_length
    # Iterate over the suction grasp path in small steps to check each point
    for step in np.arange(start_distance, end_distance, voxel_size):
        # Get current location along the suction path
        current_location = suction_location + step * suction_direction
        #print(current_location)
        # Convert current location to Vector3dVector
        query_vector = o3d.utility.Vector3dVector([current_location])
         # Create a small sphere at the query_vector location
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=voxel_size/2)
        sphere.translate(current_location)
        sphere.paint_uniform_color([1, 0, 0])  # Color the sphere red for visibility

        # Visualize the voxel grid and the sphere
        #o3d.visualization.draw_geometries([voxel_grid, sphere])

        
        # Check if the voxel at the current location is occupied
        if voxel_grid.check_if_included(query_vector)[0]:

            return True  # Collision detected

    return False  # No collision

def compute_darboux_frame(points, index, kdtree, normals):
    """
    Computes the Darboux frame for a given point.
    
    Args:
    - points (np.ndarray): Array of 3D points.
    - index (int): Index of the point to compute the frame for.
    - kdtree (o3d.geometry.KDTreeFlann): KDTree constructed using the points.
    - normals (np.ndarray): Array of normals corresponding to the points.

    Returns:
    - tuple: (t_ori, t_translate, R,normal_vector,normal_vector_opposite)
    """
    
    # Extract the point corresponding to the current index
    query = points[index,:][0:3]
    

    # Search for neighbors within a radius of 1.5 around the point
    (_, indices, _) = kdtree.search_radius_vector_3d(query,1.5)
    
    # Check if indices are out of bounds
    if np.any(np.asarray(indices) >= len(normals)):
        return None, None, None, None, None

    # Compute eigenvectors and eigenvalues for normals
    N = np.einsum('ij,ik->jk', normals[indices,:], (normals[indices,:]))               
    _, V = np.linalg.eigh(N)
    R = np.fliplr(V)
    normal_vector = R[:,0]  # Extract the normal vector
    
    # Adjust the rotation if the dot product is non-positive
    if np.dot(normal_vector, normals[index, :]) <= 0:
        R = np.matmul(R, Rz(np.pi))
        R = np.array(R)
        normal_vector = R[:, 0].reshape(3)

    # Compute the tangent vector (in this case, translation direction)
    t_ori = [query[0], query[1], query[2]]

    # Compute the transformation matrix T1
    T1 = np.eye(4)
    T1[:3, :3] = R
    T1[:3, 3] = query

    # Define a translation matrix T2 (translate 1.5cm along-x for suction location)
    T2 = np.eye(4)
    T2[0, 3] = 1.5

    # Multiply T1 and T2 to get the final transformation matrix T3
    T3 = np.matmul(T1, T2)
    t_translate = T3[:3, 3]

    normal_vector_opposite = np.array(np.matmul(R, Rz(np.pi))[:, 0]).reshape(3)
    return t_ori, t_translate,R, normal_vector,normal_vector_opposite

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def find_consecutive_rows(arr):
    consecutive_groups = []
    group_start = 0

    for index in range(1, arr.shape[0]):
        if arr[index, 9] != arr[index - 1, 9]:
            consecutive_groups.append((arr[index - 1, 9], group_start, index - 1))  # Using segmentation_id instead of group_id
            group_start = index

    consecutive_groups.append((arr[arr.shape[0] - 1, 9], group_start, arr.shape[0] - 1))  # Using segmentation_id instead of group_id
    return consecutive_groups

def Rz(theta):
    return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

def calculate_centroid(binary_mask):
    y, x = np.where(binary_mask == 1)
    centroid = (int(np.mean(y)), int(np.mean(x)))
    return centroid

def create_heatmap_centroid(dimensions, centroid, decay_factor):
    height, width = dimensions
    x, y = np.mgrid[0:height, 0:width]
    
    distances = np.sqrt((x - centroid[0])**2 + (y - centroid[1])**2)
    heatmap = np.exp(-decay_factor * distances)
    heatmap = heatmap / np.max(heatmap)
    
    return heatmap

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)

def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)
   
def load_data(image_path, pcl_path, segmentation_path):
    """Load image, point cloud, and segmentation data."""
    pcd = o3d.io.read_point_cloud(pcl_path)
    image_pil, image = load_image(image_path)
    segmentation = np.load(segmentation_path, allow_pickle=True)['arr_0']
    return pcd, image_pil, image, segmentation