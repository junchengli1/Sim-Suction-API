# Omniverse Libraries
from omni.isaac.kit import SimulationApp
from pathlib import Path
import argparse

base_dir = Path(__file__).parent
SuctionCup_path=(base_dir.parent / "Props/short_gripper.usd").as_posix()

from typing import Optional, List
import typing

# Standard Libraries
import math as m

# Third Party Libraries
import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d

import omni
import omni.physx
import omni.replicator.core as rep
import carb
from omni.isaac.core import World
from omni.isaac.debug_draw import _debug_draw
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.kit.window.viewport")  # enable legacy viewport interface
from omni.physx import get_physx_scene_query_interface
from omni.isaac.core.utils.semantics import get_semantics
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.manipulators.grippers.gripper import Gripper
from omni.isaac.surface_gripper._surface_gripper import Surface_Gripper, Surface_Gripper_Properties
from omni.isaac.universal_robots.controllers import RMPFlowController
from pxr import UsdGeom
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.manipulators.grippers.surface_gripper import SurfaceGripper

# Pixar USD Libraries
from pxr import Usd, UsdGeom, Gf, Vt

################################################################################
def gf_as_numpy(gf_matrix)->np.array:
    """Take in a pxr.Gf matrix and returns it as a numpy array.
    Specifically it transposes the matrix so that it follows numpy
    matrix rules.

    Args:
        gf_matrix (Gf.Matrix_d): Gf matrix to convert

    Returns:
        np.array:
    """
    # Convert matrix/vector to numpy array
    return np.array(list(gf_matrix))
def euler_to_rot_matrix(euler_angles: np.ndarray, degrees: bool = False) -> Gf.Rotation:
    """Convert from Euler XYZ angles to rotation matrix.

    Args:
        euler_angles (np.ndarray): Euler XYZ angles.
        degrees (bool, optional): Whether input angles are in degrees. Defaults to False.

    Returns:
        Gf.Rotation: Pxr rotation object.
    """
    return Gf.Rotation(Gf.Quatf(*euler_angles_to_quat(euler_angles, degrees)))
def get_circle_coord(theta, x_center, y_center, radius):
    """
    Computes the 3D coordinates of a point on a circle given its angle and circle parameters.
    
    Args:
    - theta (float): Angle in radians.
    - x_center (float): X-coordinate of circle's center.
    - y_center (float): Y-coordinate of circle's center.
    - radius (float): Circle's radius.

    Returns:
    - list: 3D coordinates [x, y, z] of the point on the circle.
    """
    x = 0
    y = radius * m.sin(theta) + y_center
    z=radius * m.cos(theta) + x_center
    return [x,y,z]
# This function gets all the pairs of coordinates
def get_all_circle_coords(x_center, y_center, radius, n_points):
    """
    Generates all pairs of coordinates on a circle.
    
    Args:
    - x_center (float): X-coordinate of circle's center.
    - y_center (float): Y-coordinate of circle's center.
    - radius (float): Circle's radius.
    - n_points (int): Number of points to generate on the circle.

    Returns:
    - list: A list of 3D coordinates [x, y, z] of points on the circle.
    """

    thetas = [i/n_points * m.tau for i in range(n_points)]
    circle_coords = [get_circle_coord(theta, x_center, y_center, radius) for theta in thetas]

    return circle_coords
# Using the second function to generate all the pairs of coordinates.

def get_meters_per_unit():
    """
    Fetches the meters per unit scale of the current USD stage.
    
    Returns:
    - float: Scale factor indicating meters per unit on the stage.
    """
    stage = omni.usd.get_context().get_stage()
    return UsdGeom.GetStageMetersPerUnit(stage)

def gf_as_numpy(gf_matrix)->np.array:
    """Take in a pxr.Gf matrix and returns it as a numpy array.
    Specifically it transposes the matrix so that it follows numpy
    matrix rules.

    Args:
        gf_matrix (Gf.Matrix_d): Gf matrix to convert

    Returns:
        np.array:
    """
    # Convert matrix/vector to numpy array
    return np.array(list(gf_matrix)).T
def range_with_floats(start, stop, step):
    """
    Generator that yields values in a range with float increments.
    
    Args:
    - start (float): Starting value.
    - stop (float): Stopping value.
    - step (float): Increment step.

    Yields:
    - float: Next value in the range.
    """
    while stop > start:
        yield start
        start += step

def check_raycast(origin,rayDir):
    """
    Projects a raycast from a given origin in the specified direction and checks for hits.
    
    Args:
    - origin (list): Starting point of the ray.
    - rayDir (list): Direction of the ray.

    Returns:
    - string or None: Path of the hit object if hit occurs, otherwise None.
    """
        
    # Projects a raycast from 'origin', in the direction of 'rayDir', for a length of 'distance' cm
    # Parameters can be replaced with real-time position and orientation data  (e.g. of a camera)
    origin = carb.Float3(origin[0], origin[1], origin[2])
    rayDir = carb.Float3(rayDir[0], rayDir[1], rayDir[2])
    distance = 100.0
    # physX query to detect closest hit
    hit = get_physx_scene_query_interface().raycast_closest(origin, rayDir, distance,True)
    if(hit["hit"]):
        stage = omni.usd.get_context().get_stage()
        # Change object colour to yellow and record distance from origin
        usdGeom = UsdGeom.Mesh.Get(stage, hit["rigidBody"])
        hitColor = Vt.Vec3fArray([Gf.Vec3f(255.0 / 255.0, 255.0 / 255.0, 0.0)])
        #usdGeom.GetDisplayColorAttr().Set(hitColor)
        distance = hit["distance"]
        hit_position=[hit["position"][0],hit["position"][1],hit["position"][2]]


        return usdGeom.GetPath().pathString,hit_position
    return None

def compute_suction_vertices(R, t_translate, polygon, suction_radius):
    """
    Computes the vertices of a suction cup based on a given rotation and translation.
    
    Args:
    - R (np.array): Rotation matrix.
    - t_translate (list): Translation vector.
    - polygon (int, optional): Polygonal shape for the suction. Defaults to 64.
    - suction_radius (float, optional): Radius of the suction. Defaults to 1.5.

    Returns:
    - np.array: Array of suction vertices.
    """
        
    surface_pcd = []
    for radius in range_with_floats(0.1, suction_radius, 0.1):
        circle_coords = get_all_circle_coords(x_center=0, 
                                              y_center=0,
                                              radius=radius,
                                              n_points=polygon)
        vertices = np.dot(R, np.array(circle_coords).T).T + t_translate
        surface_pcd.append(np.array(vertices))
    return np.array(surface_pcd)


def create_mesh_cylinder_detection(R, t, collision, radius=0.5, height=1):
    """
    Creates a 3D mesh of a cylinder given rotation, translation, radius, and height.

    Args:
    - R (np.array): Rotation matrix.
    - t (list): Translation vector.
    - radius (float, optional): Radius of the cylinder. Defaults to 0.1.
    - height (float, optional): Height of the cylinder. Defaults to 1.

    Returns:
    - cylinder: A 3D mesh representation of the cylinder.
    """

    cylinder = o3d.geometry.TriangleMesh().create_cylinder(radius, height)
    vertices = np.asarray(cylinder.vertices)[:, [2, 1, 0]]

    #print(np.array(vertices))

    vertices[:, 0] += height / 2
 

    vertices = np.dot(R, vertices.T).T + t
    

    cylinder.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    if collision:
        colors = np.array([0, 0, 0])
    else:
        colors = np.array([1, 1, 0])
    colors = np.expand_dims(colors, axis=0)
    colors = np.repeat(colors, vertices.shape[0], axis=0)
    cylinder.vertex_colors = o3d.utility.Vector3dVector(colors)

    return cylinder
def Rz(theta):
    return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])
def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]]) 
def Ry(theta):
  return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])

def isaac_visual(vertices, polygon,draw,dic_c_as_a_list,collision):
    """
    Visualizes given vertices in the Isaac Sim environment.

    Args:
    - vertices (list): List of 3D vertices.
    - polygon (int): Number of sides in the polygonal shape.
    - draw (object): Drawing object from Isaac Sim.
    - dic_c_as_a_list (list): List of coordinates.
    - collision (bool): Flag indicating if a collision occurred.

    Returns:
    - None
    """
    for circle_i in range(vertices.shape[0]):
        for connect_ind in range(polygon-1):
            if collision==True:
                color=[carb.Float4(1,0,0,1)]
            if collision==False:
                color=[carb.Float4(0,1,0,1)]

           
            begin=carb.Float3(np.array(vertices)[circle_i][connect_ind][0],np.array(vertices)[circle_i][connect_ind][1],np.array(vertices)[circle_i][connect_ind][2])
            end=carb.Float3(np.array(vertices)[circle_i][connect_ind+1][0],np.array(vertices)[circle_i][connect_ind+1][1],np.array(vertices)[circle_i][connect_ind+1][2])
        
            begin_after=dic_c_as_a_list[circle_i][connect_ind]
            end_after=dic_c_as_a_list[circle_i][connect_ind+1]
            

            draw.draw_lines([begin], [end], [carb.Float4(0,0,1,1)], [1])
            draw.draw_lines([begin_after], [end_after], color, [8])
            

            if connect_ind==polygon-2:

                begin_last=carb.Float3(np.array(vertices)[circle_i][0][0],np.array(vertices)[circle_i][0][1],np.array(vertices)[circle_i][0][2])
                end_last=carb.Float3(np.array(vertices)[circle_i][connect_ind+1][0],np.array(vertices)[circle_i][connect_ind+1][1],np.array(vertices)[circle_i][connect_ind+1][2])
                
                begin_after_last=dic_c_as_a_list[circle_i][0]
                end_after_last=dic_c_as_a_list[circle_i][connect_ind+1]
                
                draw.draw_lines([begin_last], [end_last], [carb.Float4(0,0,1,1)], [1])
                draw.draw_lines([begin_after_last], [end_after_last], color, [8])
            
            if (connect_ind+2)>= len(dic_c_as_a_list[circle_i]):
                break

        for connect_ind in range(len(dic_c_as_a_list[circle_i])):
                if collision==True:
                    color=[carb.Float4(1,0,0,1)]
                if collision==False:
                    color=[carb.Float4(0,1,0,1)]

                suction_before=carb.Float3(np.array(vertices)[circle_i][connect_ind][0],np.array(vertices)[circle_i][connect_ind][1],np.array(vertices)[circle_i][connect_ind][2])
                suction_after=dic_c_as_a_list[circle_i][connect_ind]
            
                draw.draw_lines([suction_before], [suction_after], [carb.Float4(1,1,1,0.2)], [1])
    
        if circle_i>0:
            
            for connect_ind in range(len(dic_c_as_a_list[circle_i])):

                if collision==True:
                    color=[carb.Float4(1,0,0,1)]
                if collision==False:
                    color=[carb.Float4(0,1,0,1)]
                draw.draw_lines([carb.Float3(np.array(vertices)[circle_i][connect_ind][0],np.array(vertices)[circle_i][connect_ind][1],np.array(vertices)[circle_i][connect_ind][2])], [carb.Float3(np.array(vertices)[circle_i-1][connect_ind][0],np.array(vertices)[circle_i-1][connect_ind][1],np.array(vertices)[circle_i-1][connect_ind][2])], [carb.Float4(0,0,1,0.2)], [1])
                if len(dic_c_as_a_list[circle_i])==len(dic_c_as_a_list[circle_i-1]):
                    draw.draw_lines([dic_c_as_a_list[circle_i][connect_ind]],[dic_c_as_a_list[circle_i-1][connect_ind]],color, [8])
                else:
                    continue
                    
    return None
        
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
    query = points[index,:]

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
    
    # Compute the binormal vector (opposite direction of normal_vector)
    normal_vector_opposite = np.array(np.matmul(R, Rz(np.pi))[:, 0]).reshape(3)

    return t_ori, t_translate,R, normal_vector,normal_vector_opposite
def calculate_distances(vertices):
    """
    Calculate the distances between consecutive vertices in a circle.

    Parameters:
    - vertices: List of vertices in a circle.

    Returns:
    - List of distances between consecutive vertices.
    """
    distances = []
    num_vertices = len(vertices)

    for i in range(num_vertices):
        next_i = (i + 1) % num_vertices
        dist = np.linalg.norm(vertices[i] - vertices[next_i])
        distances.append(dist)

    return distances

def process_vertices(vertices, deformation_distances,suction_radius):
    """
    Process a circle's vertices and deformation distances to compute original and deformed distances.

    Parameters:
    - vertices: List of original vertices.
    - deformation_distances: List of deformation distances.

    Returns:
    - Tuple containing lists of original distances, deformed distances, and deformed coordinates.
    """

    # Ensure deformation_distances contains numpy arrays
    deformation_distances = [np.array(point) for point in deformation_distances]
    #print(deformation_distances)
    # Calculate original and deformed distances
    original_dists = calculate_distances(vertices)
    deformed_dists = calculate_distances(deformation_distances)
    
    num_vertices = len(vertices)
    ver_record_after=[]
    for i in range(num_vertices):
    # Convert deformation distances to desired format (assuming carb.Float3 is desired)
        ver_record_after.append(carb.Float3(deformation_distances[i][0],deformation_distances[i][1],deformation_distances[i][2]))

        #deformed_coords = [carb.Float3(*point) for point in deformation_distances]

    return original_dists, deformed_dists, ver_record_after

def seal_evaluation(vertices, normal_vector_opposite, stage, obj_num, polygon,collision_flag,suction_radius,deformation_threshold):
    """
    Evaluates the sealing capability of the suction cup based on deformation measurements.
    
    Args:
    - vertices: The vertices to evaluate for sealing capability.
    - normal_vector_opposite: The opposite of the normal vector.
    - stage: The current simulation stage.
    - obj_num: The object number being evaluated.
    - polygon: The polygonal shape for the suction cup.
    - collision_flag: Flag indicating if there's a collision.
    
    Returns:
    - collision_flag: Updated flag indicating if there's a collision after evaluation.
    - deformation_measurements: Measurements indicating deformation.
    """
    deformation_measurements = {}
    for circle_idx in range(vertices.shape[0]):

        deformation_distances = []
        current_circle_vertices = np.array(vertices)[circle_idx]
        # Check for collisions and record distances for deformation calculations
        for vertex in current_circle_vertices:
            if (check_raycast(vertex, normal_vector_opposite)) == None:
                collision_flag = True
                position=[10000,10000,10000]
            else:
                collided_prim, position = check_raycast(vertex, normal_vector_opposite)
                collision_prims_list=[]

                if obj_num!=None:
                    curr_prim = stage.GetPrimAtPath(f"/World/objects/object_{obj_num-1}")

                    for prim in Usd.PrimRange(curr_prim):
                        if (prim.IsA(UsdGeom.Xformable)):
                            collision_prims_list.append(prim.GetPath().pathString)
                    
                    if collided_prim not in collision_prims_list or collided_prim in ["/World/groundPlane/collisionPlane", "/World/groundPlane/geom"]:
                        collision_flag = True
                

            deformation_distances.append(position)

        # Calculate deformation by comparing distances between vertices
        original_dists, deformed_dists, deformed_coords = process_vertices(current_circle_vertices, deformation_distances,suction_radius)
        deformation_measurements[circle_idx] = deformed_coords

        # Calculate deformation thresholds
        if len(deformed_dists) < polygon:
            deformation_thresholds = np.full((polygon, 1), 100)
        else:
            deformation_thresholds = abs(np.array(deformed_dists) - np.array(original_dists)) / original_dists
        
        # Check if deformation is beyond the threshold
        if np.any(deformation_thresholds >= deformation_threshold):
            collision_flag = True

        # Visualization: Create point cloud for the current circle
        #draw_vertices_pcd = o3d.geometry.PointCloud()
        #draw_vertices_pcd.points = o3d.utility.Vector3dVector(current_circle_vertices)
    
    return collision_flag,deformation_measurements

def get_camera_info():
    """
    Sets up the camera information for the simulation.
    3-5 cameras are recommended due to memory issue.
    or use one random camera position and look_at (0,0,0) and merge single shot point clouds (used in the manuscript/slow).
    Returns:
    - pcd_annotators: List of point cloud annotators attached to the render products.
    - sem_annots: List of semantic segmentation annotators attached to the render products.
    """
    # Setting up cameras and render products
    W, H = (1280, 720)
    camera = rep.create.camera(position=(0, 0, 220), look_at=(0,0,0))
    camera1 = rep.create.camera(position=(0, -300, 150), look_at=(0,0,0))
    camera2 = rep.create.camera(position=(0, 300, 150), look_at=(0,0,0))
    camera3 = rep.create.camera(position=(300, 0, 150), look_at=(0,0,0))
    camera4 = rep.create.camera(position=(-300, 0, 150), look_at=(0,0,0))

    rp = rep.create.render_product(camera, (W, H))
    rp1 = rep.create.render_product(camera1, (W, H))
    rp2 = rep.create.render_product(camera2, (W, H))
    rp3 = rep.create.render_product(camera3, (W, H))
    rp4 = rep.create.render_product(camera4, (W, H))    
    pcd_annotators=[]
    sem_annots=[]
    # Attach point cloud and semantic segmentation annotators to render products

    for render_product in [rp, rp1, rp2,rp3,rp4]:
        pointcloud_anno = rep.annotators.get("pointcloud")
        semantic_segmentation_anno = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
        pointcloud_anno.attach([render_product])
        semantic_segmentation_anno.attach([render_product])
        pcd_annotators.append(pointcloud_anno)
        sem_annots.append(semantic_segmentation_anno)
    return pcd_annotators,sem_annots

def get_semantic_info(pcd_annotators,sem_annots):
    """
    Extracts the semantic information from the simulation's point clouds.
    
    Args:
    - pcd_annotators: List of point cloud annotators.
    - sem_annots: List of semantic segmentation annotators.
    
    Returns:
    - all_unique_classes: Dictionary of unique classes detected in the simulation.
    - label_to_object_number_map: Mapping of labels to object numbers.
    """
    # Extract semantic information
    stage = omni.usd.get_context().get_stage()
    label_to_object_number_map = {}
    curr_prim = stage.GetPrimAtPath("/World/objects")
    children_list=curr_prim.GetChildren()
    for prim in children_list:
        prim_path = str(prim.GetPath())
        # Extract object number from path
        object_number = int(prim_path.split('_')[-1])  # Assuming the path always has the format "/World/objects/object_X"
        prim_sd = get_semantics(prim)
        label = prim_sd.get('Semantics', (None, None))[1]
        if label:
            label_to_object_number_map[label.lower()] = object_number+1
    label_to_object_number_map["bin"] = 0

    rep.orchestrator.step()
    all_unique_classes = {} 

    # Extract unique instances and classes from each camera's point cloud
    for j, pcd_annot in enumerate(pcd_annotators):
        pc_data = pcd_annot.get_data()
        segmentation_pcd = pc_data["info"]["pointSemantic"]
        unique_instances = np.unique(segmentation_pcd)
        sem_data = sem_annots[j].get_data()  # Change indexing to j from k
        id_to_labels = sem_data["info"]["idToLabels"]
        for instance in unique_instances:
            if str(instance) in id_to_labels:
                class_name = id_to_labels[str(instance)]['class']
                if class_name not in all_unique_classes:
                    all_unique_classes[class_name] = []
                all_unique_classes[class_name].append((j, instance))  # Store camera obj_num along with instance obj_num
    return all_unique_classes, label_to_object_number_map

def get_merge_pointcloud(all_unique_classes,pcd_annotators,label_to_object_number_map,stage_folder,save):
    """
    Merges point clouds from all cameras for each unique class.
    
    Args:
    - all_unique_classes: Dictionary of unique classes.
    - pcd_annotators: List of point cloud annotators.
    - label_to_object_number_map: Mapping of labels to object numbers.
    - stage_folder: Path to the stage folder.
    - save: Boolean indicating whether to save the merged point cloud to disk.
    
    Returns:
    - point_cloud_dic: Dictionary containing merged point clouds for each unique class.
    - all_instance_seg_ids: List of segmentation IDs for all instances.
    """
    all_instance_seg_ids=[]
    point_cloud_dic={}
    for class_name, instance_ids in all_unique_classes.items():
        class_pcds = []
        # For each point cloud annotator (camera)...
        for (camera_id, instance_id) in instance_ids:  # Unpack camera obj_num and instance obj_num
            pcd_annot = pcd_annotators[camera_id]  # Get pcd_annot based on camera obj_num
            pc_data = pcd_annot.get_data()
            points = pc_data["data"]
            normals = pc_data["info"]["pointNormals"].reshape(-1, 4)[:, :3]
            colors = ((pc_data["info"]["pointRgb"])/255).reshape(-1, 4)[:, :3]
            segmentation_pcd = pc_data["info"]["pointSemantic"]
            # Filter points, normals, and colors based on the instance obj_num
            instance_points = points[segmentation_pcd == instance_id]
            instance_normals = normals[segmentation_pcd == instance_id]
            instance_colors = colors[segmentation_pcd == instance_id]

            # Create a PointCloud object
            pcd = o3d.geometry.PointCloud()
            # Assign the points, normals, and colors to the PointCloud object
            pcd.points = o3d.utility.Vector3dVector(instance_points)
            pcd.normals = o3d.utility.Vector3dVector(instance_normals)
            pcd.colors = o3d.utility.Vector3dVector(instance_colors)

            # Add the PointCloud object to the list
            class_pcds.append(pcd)

        # Merge point clouds of this instance class from all cameras
        merged_pcd = class_pcds[0]
        for pcd in class_pcds[1:]:
            merged_pcd += pcd

        # Add the merged PointCloud object to the list
        if class_name=="UNLABELLED":
            class_name="ground"
            label_to_object_number_map[class_name]=0
        if save==True:
            o3d.io.write_point_cloud(stage_folder+"/" +class_name +"_"+str(label_to_object_number_map[class_name])+".pcd",merged_pcd)
        all_instance_seg_ids.append(int(label_to_object_number_map[class_name]))
        point_cloud_dic[int(label_to_object_number_map[class_name])]=merged_pcd
        points=np.asarray(merged_pcd.points)
    return point_cloud_dic,all_instance_seg_ids



def hasSchema(prim, schemaName):
    schemas = prim.GetAppliedSchemas()
    for s in schemas:
        if s == schemaName:
            return True
    return False


class SimSuctionController_base(BaseController):
    """
    A suction pick state machine.

    The state machine goes through multiple phases:
    - Phase 0: Move the end effector above the object's center at the specified height.
    - Phase 1: Lower the end effector to encircle the target object.
    - Phase 2: Wait for the Robot's inertia to settle.
    - Phase 3: Close the grip.
    - Phase 4: Raise the end effector, lifting the object.
    - Phase 5: Move the end effector towards the goal position, maintaining height.

    Args:
        name (str): Name of the controller.
        cspace_controller (BaseController): Cartesian space controller returning an ArticulationAction type.
        gripper (Gripper): Gripper controller for grip actions.
        end_effector_initial_height (float, optional): Initial height to start from. Defaults to 0.3 meters.
        events_dt (List[float], optional): Time deltas for each phase. 6 values must be defined.

    Raises:
        Exception: If `events_dt` is not a list or numpy array.
        Exception: If `events_dt` does not have a length of 6.
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0
        self._h1 = end_effector_initial_height
        if self._h1 is None:
            self._h1 = 0.3 / get_stage_units()
        self._h0 = None
        self._events_dt = events_dt
        if self._events_dt is None:
            self._events_dt = [0.008, 0.005, 0.1, 0.1, 0.0025, 1]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt)  != 6:
                raise Exception("events dt length must be less than 6")
        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._pause = False
        return

    def is_paused(self) -> bool:
        """

        Returns:
            bool: True if the state machine is paused. Otherwise False.
        """
        return self._pause

    def get_current_event(self) -> int:
        """

        Returns:
            int: Current event/ phase of the state machine
        """
        return self._event

    def forward(
        self,
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """Runs the controller one step.

        Args:
            picking_position (np.ndarray): The object's position to be picked in local frame.
            placing_position (np.ndarray):  The object's position to be placed in local frame.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            end_effector_offset (typing.Optional[np.ndarray], optional): offset of the end effector target. Defaults to None.
            end_effector_orientation (typing.Optional[np.ndarray], optional): end effector orientation while picking and placing. Defaults to None.

        Returns:
            ArticulationAction: action to be executed by the ArticulationController
        """
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0])
        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        if self._event == 2:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 3:
            target_joint_positions = self._gripper.forward(action="close")
        else:
            if self._event in [0, 1]:
                self._current_target_x = picking_position[0]
                self._current_target_y = picking_position[1]
                self._h0 = picking_position[2]
            interpolated_xy = self._get_interpolated_xy(
                placing_position[0], placing_position[1], self._current_target_x, self._current_target_y
            )
            target_height = self._get_target_hs(placing_position[2])
            position_target = np.array(
                [
                    interpolated_xy[0] + end_effector_offset[0],
                    interpolated_xy[1] + end_effector_offset[1],
                    target_height + end_effector_offset[2],
                ]
            )
            if end_effector_orientation is None:
                end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target, target_end_effector_orientation=end_effector_orientation
            )
        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0
        return target_joint_positions

    def _get_interpolated_xy(self, target_x, target_y, current_x, current_y):
        alpha = self._get_alpha()
        xy_target = (1 - alpha) * np.array([current_x, current_y]) + alpha * np.array([target_x, target_y])
        return xy_target

    def _get_alpha(self):
        if self._event < 5:
            return 0
        elif self._event == 5:
            return self._mix_sin(self._t)
        else:
            raise ValueError()

    def _get_target_hs(self, target_height):
        if self._event == 0:
            h = self._h1
        elif self._event == 1:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h1, self._h0, a)
        elif self._event == 3:
            h = self._h0
        elif self._event == 4:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h0, self._h1, a)
        elif self._event == 5:
            h = self._h1
        else:
            raise ValueError()
        return h

    def _mix_sin(self, t):
        return 0.5 * (1 - np.cos(t * np.pi))

    def _combine_convex(self, a, b, alpha):
        return (1 - alpha) * a + alpha * b

    def reset(
        self,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        """Resets the state machine to start from the first phase/ event

        Args:
            end_effector_initial_height (typing.Optional[float], optional): end effector initial picking height to start from. If not defined, set to 0.3 meters. Defaults to None.
            events_dt (typing.Optional[typing.List[float]], optional):  Dt of each phase/ event step. 6 phases dt has to be defined. Defaults to None.

        Raises:
            Exception: events dt need to be list or numpy array
            Exception: events dt need have length of 6
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        if end_effector_initial_height is not None:
            self._h1 = end_effector_initial_height
        self._pause = False
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) or not isinstance(self._events_dt, list):
                raise Exception("event velocities need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) == 6:
                raise Exception("events dt length must be less than 10")
        return

    def is_done(self) -> bool:
        """
        Returns:
            bool: True if the state machine reached the last phase. Otherwise False.
        """
        if self._event >= len(self._events_dt):
            return True
        else:
            return False

    def pause(self) -> None:
        """Pauses the state machine's time and phase.
        """
        self._pause = True
        return

    def resume(self) -> None:
        """Resumes the state machine's time and phase.
        """
        self._pause = False
        return       
class PickPlaceController(SimSuctionController_base):
    """[summary]

        Args:
            name (str): [description]
            surface_gripper (SurfaceGripper): [description]
            robot_articulation(Articulation): [description]
            events_dt (Optional[List[float]], optional): [description]. Defaults to None.
        """

    def __init__(
        self,
        name: str,
        gripper: SurfaceGripper,
        robot_articulation: Articulation,
        events_dt: Optional[List[float]] = None,
    ) -> None:
        if events_dt is None:
            events_dt = None
        SimSuctionController_base.__init__(
            self,
            name=name,
            cspace_controller=RMPFlowController(
                name=name + "_cspace_controller", robot_articulation=robot_articulation, attach_gripper=True
            ),
            gripper=gripper,
            events_dt=events_dt,
        )
        return

    def forward(
        self,
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: Optional[np.ndarray] = None,
        end_effector_orientation: Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """[summary]

        Args:
            picking_position (np.ndarray): [description]
            placing_position (np.ndarray): [description]
            current_joint_positions (np.ndarray): [description]
            end_effector_offset (Optional[np.ndarray], optional): [description]. Defaults to None.
            end_effector_orientation (Optional[np.ndarray], optional): [description]. Defaults to None.

        Returns:
            ArticulationAction: [description]
        """
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi / 2.0, 0]))
        return super().forward(
            picking_position,
            placing_position,
            current_joint_positions,
            end_effector_offset=end_effector_offset,
            end_effector_orientation=end_effector_orientation,
        )
class SurfaceGripper(Gripper):
    """Provides high level functions to set/ get properties and actions of a surface gripper 
        (a suction cup for example).

        Args:
            end_effector_prim_path (str): prim path of the Prim that corresponds to the gripper root/ end effector.
            translate (float, optional): _description_. Defaults to 0.
            direction (str, optional): _description_. Defaults to "x".
            grip_threshold (float, optional): _description_. Defaults to 0.01.
            force_limit (float, optional): _description_. Defaults to 1.0e6.
            torque_limit (float, optional): _description_. Defaults to 1.0e4.
            bend_angle (float, optional): _description_. Defaults to np.pi/24.
            kp (float, optional): _description_. Defaults to 1.0e2.
            kd (float, optional): _description_. Defaults to 1.0e2.
            disable_gravity (bool, optional): _description_. Defaults to True.
        """

    def __init__(
        self,
        end_effector_prim_path: str,
        translate: float = 0,
        direction: str = "x",
        grip_threshold: float = 1,  
        force_limit: float = 3000,  ###### newton*100
        torque_limit: float = 100000,
        bend_angle: float = np.pi /24,   
        kp: float = 1.0e4,
        kd: float = 1.0e3, 
        disable_gravity: bool = False,
    ) -> None:
        Gripper.__init__(self, end_effector_prim_path=end_effector_prim_path)
        self._dc_interface = _dynamic_control.acquire_dynamic_control_interface()
        self._translate = translate
        self._direction = direction
        self._grip_threshold = grip_threshold
        self._force_limit = force_limit
        self._torque_limit = torque_limit
        self._bend_angle = bend_angle
        self._kp = kp
        self._kd = kd
        self._disable_gravity = disable_gravity
        self._virtual_gripper = None
        self._articulation_num_dofs = None
        return

    def initialize(
        self, physics_sim_view: omni.physics.tensors.SimulationView = None, articulation_num_dofs: int = None
    ) -> None:
        """Create a physics simulation view if not passed and creates a rigid prim view using physX tensor api.
            This needs to be called after each hard reset (i.e stop + play on the timeline) before interacting with any
            of the functions of this class.

        Args:
            physics_sim_view (omni.physics.tensors.SimulationView, optional): current physics simulation view. Defaults to None
            articulation_num_dofs (int, optional): num of dofs of the Articulation. Defaults to None.
        """
        Gripper.initialize(self, physics_sim_view=physics_sim_view)
        self._articulation_num_dofs = articulation_num_dofs
        virtual_gripper_props = Surface_Gripper_Properties()
        virtual_gripper_props.parentPath = self._end_effector_prim_path
        virtual_gripper_props.d6JointPath = virtual_gripper_props.parentPath + "/d6FixedJoint"
        virtual_gripper_props.gripThreshold = self._grip_threshold
        virtual_gripper_props.forceLimit = self._force_limit
        virtual_gripper_props.torqueLimit = self._torque_limit
        virtual_gripper_props.bendAngle = self._bend_angle
        virtual_gripper_props.stiffness = self._kp
        virtual_gripper_props.damping = self._kd
        virtual_gripper_props.disableGravity = self._disable_gravity
        tr = _dynamic_control.Transform()
        if self._direction == "x":
            tr.p.x = self._translate
        elif self._direction == "y":
            tr.p.y = self._translate
        elif self._direction == "z":
            tr.p.z = self._translate
        else:
            carb.log_error("Direction specified for the surface gripper doesn't exist")
        virtual_gripper_props.offset = tr
        virtual_gripper = Surface_Gripper(self._dc_interface)
        virtual_gripper.initialize(virtual_gripper_props)
        self._virtual_gripper = virtual_gripper
        if self._default_state is None:
            self._default_state = not self.is_closed()
        return

    def close(self) -> None:
        """Applies actions to the articulation that closes the gripper (ex: to hold an object).
        """
        if not self.is_closed():
            self._virtual_gripper.close()
        if not self.is_closed():
            carb.log_warn("close suction...")
        return

    def open(self) -> None:
        """Applies actions to the articulation that opens the gripper (ex: to release an object held).
        """
        result = self._virtual_gripper.open()
        if not result:
            carb.log_warn("open suction...")

        return

    def update(self) -> None:
        self._virtual_gripper.update()
        return

    def is_closed(self) -> bool:
        return self._virtual_gripper.is_closed()

    def set_translate(self, value: float) -> None:
        self._translate = value
        return

    def set_direction(self, value: float) -> None:
        self._direction = value
        return

    def set_force_limit(self, value: float) -> None:
        self._force_limit = value
        return

    def set_torque_limit(self, value: float) -> None:
        self._torque_limit = value
        return

    def set_default_state(self, opened: bool):
        """Sets the default state of the gripper

        Args:
            opened (bool): True if the surface gripper should start in an opened state. False otherwise.
        """
        self._default_state = opened
        return

    def get_default_state(self) -> dict:
        """Gets the default state of the gripper

        Returns:
            dict: key is "opened" and value would be true if the surface gripper should start in an opened state. False otherwise.
        """
        return {"opened": self._default_state}

    def post_reset(self):
        Gripper.post_reset(self)
        if self._default_state:  # means opened is true
            self.open()
        else:
            self.close()
        return

    def forward(self, action: str) -> ArticulationAction:
        """calculates the ArticulationAction for all of the articulation joints that corresponds to "open"
           or "close" actions.

        Args:
            action (str): "open" or "close" as an abstract action.

        Raises:
            Exception: _description_

        Returns:
            ArticulationAction: articulation action to be passed to the articulation itself
                                (includes all joints of the articulation).
        """
        if self._articulation_num_dofs is None:
            raise Exception(
                "Num of dofs of the articulation needs to be passed to initialize in order to use this method"
            )
        if action == "open":
            self.open()
        elif action == "close":
            self.close()
        else:
            raise Exception("action {} is not defined for SurfaceGripper".format(action))
        return ArticulationAction(joint_positions=[None] * self._articulation_num_dofs)
class Robot_arm(Robot):
    """[summary]

        Args:
            prim_path (str): [description]
            name (str, optional): [description]. Defaults to "ur10_robot".
            usd_path (Optional[str], optional): [description]. Defaults to None.
            position (Optional[np.ndarray], optional): [description]. Defaults to None.
            orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
            end_effector_prim_name (Optional[str], optional): [description]. Defaults to None.
            attach_gripper (bool, optional): [description]. Defaults to False.
            gripper_usd (Optional[str], optional): [description]. Defaults to "default".

        Raises:
            NotImplementedError: [description]
        """

    def __init__(
        self,
        prim_path: str,
        name: str = "Robot_arm",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        end_effector_prim_name: Optional[str] = None,
        attach_gripper: bool = False,
        gripper_usd: Optional[str] = "default",
    ) -> None:
        prim = get_prim_at_path(prim_path)
        self._end_effector = None
        self._gripper = None
        self._end_effector_prim_name = end_effector_prim_name
        if not prim.IsValid():
            if usd_path:
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            else:
                assets_root_path = get_assets_root_path()
                if assets_root_path is None:
                    carb.log_error("Could not find Isaac Sim assets folder")
                    return
                usd_path = assets_root_path + "/Isaac/Robots/UR10/ur10.usd"
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            if self._end_effector_prim_name is None:
                self._end_effector_prim_path = prim_path + "/ee_link"
            else:
                self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
        else:
            if self._end_effector_prim_name is None:
                self._end_effector_prim_path = prim_path + "/ee_link"
            else:
                self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
        super().__init__(
            prim_path=prim_path, name=name, position=position, orientation=orientation, articulation_controller=None
        )
        self._gripper_usd = gripper_usd
        if attach_gripper:
            if gripper_usd == "default":
                assets_root_path = get_assets_root_path()
                if assets_root_path is None:
                    carb.log_error("Could not find Isaac Sim assets folder")
                    return
                gripper_usd=SuctionCup_path
                add_reference_to_stage(usd_path=gripper_usd, prim_path=self._end_effector_prim_path)
                self._gripper = SurfaceGripper(
                    end_effector_prim_path=self._end_effector_prim_path, translate=16.12, direction="x"
                )
            elif gripper_usd is None:
                carb.log_warn("Not adding a gripper usd, the gripper already exists in the Robot Arm asset")
                self._gripper = SurfaceGripper(
                    end_effector_prim_path=self._end_effector_prim_path, translate=16.12, direction="x"
                )
            else:
                raise NotImplementedError
        self._attach_gripper = attach_gripper
        return

    @property
    def attach_gripper(self) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        return self._attach_gripper

    @property
    def end_effector(self) -> RigidPrim:
        """[summary]

        Returns:
            RigidPrim: [description]
        """
        return self._end_effector

    @property
    def gripper(self) -> SurfaceGripper:
        """[summary]

        Returns:
            SurfaceGripper: [description]
        """
        return self._gripper

    def initialize(self, physics_sim_view=None) -> None:
        """[summary]
        """
        super().initialize(physics_sim_view)
        if self._attach_gripper:
            self._gripper.initialize(physics_sim_view=physics_sim_view, articulation_num_dofs=self.num_dof)
        self._end_effector = RigidPrim(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        self.disable_gravity()
        self._end_effector.initialize(physics_sim_view)
        return

    def post_reset(self) -> None:
        Robot.post_reset(self)
        self._end_effector.post_reset()
        self._gripper.post_reset()
        return










