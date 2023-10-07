import argparse
import os
import pickle
import sys
current_directory = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_directory)
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import open3d as o3d
from sim_suction_model.sim_suction_pointnet import ScoreNet
import math as m
import copy
import matplotlib.pyplot as plt
from pathlib import Path

base_dir = Path(__file__).parent
parser = argparse.ArgumentParser(description='inference')
parser.add_argument('--pointcloud_test_path', type=str, default="../test_novel_pointcloud", choices=["../test_similar_pointcloud","../test_novel_pointcloud"],help='point cloud path')
parser.add_argument('--save_result_path', type=str, default="../test_novel_inference_result", choices=["../test_similar_inference_result","../test_novel_inference_result"], help='save result path')
parser.add_argument('--total_frame', type=int, default=100, help='total frame number')
parser.add_argument('--total_stage', type=int, default=100, help='total stage number')
parser.add_argument('--model_name', type=str, default="/score_99.model", help='model name')
parser.add_argument('--model_path', type=str, default= (base_dir / 'models').as_posix(), help='saved model path')
parser.add_argument('--visualize', type=str, default="top_1%", choices=["top_1", "top_1%", "top_5%", "top_10%"],
                    help="Which subset to visualize (e.g., top_1, top_1%, etc.)")
parser.add_argument('--visualization_flag', type=bool, default=False, help='visualization flag')
parser.add_argument('--save_result_flag', type=bool, default=True, help='save inference result flag')

args = parser.parse_args()

######################################################################################################3
score_model = ScoreNet(training=False).cuda()
resume_epoch=0
model_path=args.model_path+args.model_name
model_dict = torch.load(model_path, map_location='cuda:{}'.format(0)).state_dict() #, map_location='cpu'
new_model_dict = {}
for key in model_dict.keys():
     new_model_dict[key.replace("module.", "")] = model_dict[key]
score_model.load_state_dict(new_model_dict)

#####################################################################################################3
def create_mesh_cylinder_detection(R, t, collision, radius=0.5, height=5):
    cylinder = o3d.geometry.TriangleMesh().create_cylinder(radius, height)
    vertices = np.asarray(cylinder.vertices)[:, [2, 1, 0]]
    vertices[:, 0] += height / 2
    vertices = np.dot(R, vertices.T).T + t
    
    cylinder.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    if collision:
        colors = np.array([1, 0, 0])
    else:
        colors = np.array([0, 1, 0])
    colors = np.expand_dims(colors, axis=0)
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


def inference_one_scene(data_root, stage_ind):
    points=[]
    points_ori=[]
    pc=[]
    score_model.eval()
    with torch.no_grad():
        for frame in range(0,args.total_frame):
            display=[]

            print(f'---- Inference {args.pointcloud_test_path}, stage: {stage_ind}, frame: {frame} ----')
            print(f'Save result flag: {args.save_result_flag}')
            print(f'Visualization flag: {args.visualization_flag}')

            points=np.load(data_root+f"/{stage_ind}_{frame}.npz",allow_pickle=True)['arr_0']
            #points=np.load(data_root+f"/{stage_ind}.npz",allow_pickle=True)['data']
            
            point_cloud=[]
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points[:,:3]) 
            point_cloud.normals = o3d.utility.Vector3dVector(points[:,3:6]) 
            point_cloud.colors = o3d.utility.Vector3dVector(points[:,6:9])

            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, 1)

            points_ori=copy.deepcopy(points)
            

            points=points[:, 0:6]
            points[:, 0:3] = pc_normalize(points[:, 0:3])

            if len(points) < 5120:
                # Calculate the number of points needed
                num_points_needed = 5120 - len(points)           
                zero_padding = np.zeros((num_points_needed, points.shape[1]))
                points = np.vstack((points, zero_padding))

            points=torch.from_numpy(points).float(). unsqueeze(0)
        
        
            pc = points.cuda()
            
            results,_ = score_model(pc, None)

            results_dict = {
                "top_1": {},
                "top_1%": {},
                "top_5%": {},
                "top_10%": {}
            }

            if results is None:
                print("No suction grasp candidates detected.")
                continue
        
            top_10_indices = results['top_10%'][1]
            ind_list,t_ori_list, t_translate_list, R_list, normal_vector_list, normal_vector_opposite_list = [],[], [], [], [], []


            normals=np.asarray(point_cloud.normals)
            kdtree = o3d.geometry.KDTreeFlann(point_cloud)

            for index in top_10_indices:
                t_ori, t_translate, R, normal_vector, normal_vector_opposite = compute_darboux_frame(points_ori, index, kdtree, normals)

                # Check for collision
                collision = check_collision_with_suction_gripper_using_voxelgrid(voxel_grid, t_translate, normal_vector)
                # If no collision, append to lists
                if not collision:
                    t_ori_list.append(t_ori)
                    t_translate_list.append(t_translate)
                    R_list.append(R)
                    normal_vector_list.append(normal_vector)
                    normal_vector_opposite_list.append(normal_vector_opposite)
                    ind_list.append(index)
            
            if ind_list==[]:
                print("No collision-free suction grasp candidates detected.")
                continue

            results_dict["top_10%"] = {
                "t_ori": t_ori_list,
                "t_translate": t_translate_list,
                "R": R_list,
                "normal_vector": normal_vector_list,
                "normal_vector_opposite": normal_vector_opposite_list,
                "relative_indices": list(range(len(ind_list)))
            }
            for top_key in ["top_1", "top_1%", "top_5%"]:
                absolute_indices = results[top_key][1]
                relative_indices = [top_10_indices.index(idx) for idx in absolute_indices if idx in ind_list]
                results_dict[top_key]["relative_indices"] = relative_indices
           
            # Extract specified subset's relative indices
            subset_relative_indices = results_dict[args.visualize]["relative_indices"]

            if args.save_result_flag == True:
                if not os.path.exists(args.save_result_path):
                    os.makedirs(args.save_result_path)
                save_path = os.path.join(args.save_result_path, f"stage_{stage_ind}_{frame}_results.pkl")
                print(save_path)
                with open(save_path, "wb") as f:
                    pickle.dump(results_dict, f)


            if args.visualization_flag ==True:
                # Get data for the specified subset from top_10%
                subset_data = {
                    "t_ori": [results_dict["top_10%"]["t_ori"][idx] for idx in subset_relative_indices],
                    "t_translate": [results_dict["top_10%"]["t_translate"][idx] for idx in subset_relative_indices],
                    "R": [results_dict["top_10%"]["R"][idx] for idx in subset_relative_indices],
                    "normal_vector": [results_dict["top_10%"]["normal_vector"][idx] for idx in subset_relative_indices],
                    "normal_vector_opposite": [results_dict["top_10%"]["normal_vector_opposite"][idx] for idx in subset_relative_indices],
                }
                # Visualize the specified subset's results
                for i in range(len(subset_relative_indices)):
                    t_ori = subset_data["t_ori"][i]
                    t_translate = subset_data["t_translate"][i]
                    R = subset_data["R"][i]
                    normal_vector =subset_data["normal_vector"][i]
                    normal_vector_opposite = subset_data["normal_vector_opposite"][i]
                    # Create mesh and add to the display list
                    mesh = create_mesh_cylinder_detection(R, t_ori, False)
                    display.append(mesh)


                top_scores = results[args.visualize][0]
                top_percent_indices= results[args.visualize][1]
                colors = np.asarray(point_cloud.colors)

                colormap = plt.get_cmap("viridis")

                heatmap_colors = colormap(top_scores)[:,0:3]

                new_colors = np.array(colors)

                new_colors[top_percent_indices] = heatmap_colors
        
                point_cloud.colors = o3d.utility.Vector3dVector(new_colors)

                display.append(point_cloud)

                o3d.visualization.draw_geometries_with_custom_animation(display,width=720,height=720)




def inference(stage_ind):
          
    pcl_dir=args.pointcloud_test_path
    inference_one_scene(pcl_dir,stage_ind)



if __name__ == "__main__":

    for stage_ind in range(0,args.total_stage):
        inference(stage_ind)
    

