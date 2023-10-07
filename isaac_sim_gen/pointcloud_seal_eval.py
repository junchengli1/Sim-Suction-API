# Omniverse Libraries
from omni.isaac.kit import SimulationApp
from pathlib import Path
import argparse
base_dir = Path(__file__).parent
parser = argparse.ArgumentParser(description='Sim-Suction create_point_cloud_and_seal_evaluation')
parser.add_argument('--headless', type=bool, default=False, help='headless')#make it True when gerneate dataset 
parser.add_argument('--debug_draw', type=bool, default=True, help='debug draw')#make it False when gerneate dataset 
parser.add_argument('--pcl_path', type=str, default=(base_dir.parent / "pointcloud_train").as_posix(), help='point cloud path')
parser.add_argument('--data_path', type=str, default=(base_dir.parent / "synthetic_data").as_posix(), help='data path')
parser.add_argument('--instanceable_flag', type=bool, default=True, help='use textureless instanceable usd to increase simulation speed')
parser.add_argument('--seg_dic_path', type=str, default=(base_dir.parent / "seg_dic.pkl").as_posix(), help='seg_dic path')
parser.add_argument('--save_pcd_flag', type=bool, default=False, help='Save each objects as pcd file')
parser.add_argument('--save_pkl_flag', type=bool, default=False, help='Save seal evaluation as pkl file')
parser.add_argument('--save_pcl_flag', type=bool, default=False, help='Save merged point cloud as npz file')
parser.add_argument('--suction_radius', type=float, default=1.5, help='suction radius 1.5cm')
parser.add_argument('--deformation_threshold', type=float, default=0.15, help='15 percent deformation')
parser.add_argument('--start_stage', type=int,default=0, help='start stage number')
parser.add_argument('--end_stage', type=int,default=500, help='end stage number')


args = parser.parse_args()
simulation_app = SimulationApp({"headless": args.headless})

from omni.isaac.core import World
from omni.isaac.debug_draw import _debug_draw
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.kit.window.viewport")  # enable legacy viewport interface

from omni.physx.scripts import utils
import omni
import omni.physx
import omni.replicator.core as rep
import carb
   
# Pixar USD Libraries
from pxr import Usd, UsdGeom

# Standard Libraries
import os
import pickle
import math as m
import asyncio

# Third Party Libraries
import numpy as np
import open3d as o3d
import torch
import torch.utils.data

from dgl.geometry import farthest_point_sampler

from simulation_utils import *  # This imports all functions from the utility file


class generate_pcl_and_seal():
    def __init__(self, data_root):
        self.data_root = data_root
        self.pointcloud_root= args.pcl_path
        self.stage_root = os.listdir(data_root)
        with open(args.seg_dic_path, "rb") as f:
                seg_dic=pickle.load(f)
        f.close()
        self.seg_dic=seg_dic


    async def load_stage(self, path):
        await omni.usd.get_context().open_stage_async(path)
    
    def __getitem__(self, stage_ind,save_pcl_flag,save_seal_flag):
        # Load the stage asynchronously
        stage_folder=self.data_root+f"/stage_{stage_ind}"
        if args.instanceable_flag==True:
            stage_usd_list=self.data_root+f"/stage_{stage_ind}"+ f"/stage_{stage_ind}_instanceable.usd"
        else:
            stage_usd_list=self.data_root+f"/stage_{stage_ind}"+ f"/stage_{stage_ind}.usd"
        setup_task = asyncio.ensure_future(self.load_stage(stage_usd_list))

        while not setup_task.done():
            simulation_app.update()

        # Create World
        the_world = World(stage_units_in_meters = 0.01)
        
        if args.instanceable_flag==True:
            # Create a ground plane
            the_world.scene.add_ground_plane(size=1000, color=np.array([1, 1, 1]))

        stage = omni.usd.get_context().get_stage()
         
        pcd_annotators,sem_annots=get_camera_info()
            
        # Update and play the simulation

        # Remove rigid bodies and set colliders for all mesh primitives
        # Rigid body will use convex hull as collider default which is not accurate
 
        simulation_app.update()
        the_world.play()

        for i in range(150):
            the_world.step(render=True)

        curr_prim = stage.GetPrimAtPath("/World/objects")
        for prim in Usd.PrimRange(curr_prim):
            if prim.IsA(UsdGeom.Xform):
                utils.removeRigidBody(prim)
            elif prim.IsA(UsdGeom.Mesh):
                utils.setCollider(prim, approximationShape="none")

     
        all_unique_classes, label_to_object_number_map=get_semantic_info(pcd_annotators,sem_annots)
    
        # For each unique class, combine point clouds from all cameras, and perform raycasting and collision detection
        point_cloud_dic,all_instance_seg_ids=get_merge_pointcloud(all_unique_classes,pcd_annotators,label_to_object_number_map,stage_folder,args.save_pcd_flag)
       
        draw = _debug_draw.acquire_debug_draw_interface()

        draw.clear_points()
        draw.clear_lines()
        
        # Create a map for object number to label
        object_number_to_label_map = {v: k for k, v in label_to_object_number_map.items()}
       
        # Initial lists and dictionaries
        display=[]
        candidates={}

        # Placeholder array for storing point clouds
        point_clouds_stage=np.empty((0,10), float)

        # Iterate over all unique segment ids
        print(all_instance_seg_ids)

        for obj_num in all_instance_seg_ids:
            class_name=object_number_to_label_map[obj_num]
            print(object_number_to_label_map[obj_num])
            # If current instance is ground
            if obj_num==0:
                
                pcd=point_cloud_dic[obj_num]
                display.append(pcd)
                points=np.asarray(pcd.points)
                normals=np.asarray(pcd.normals)
                segmentation_ground=np.full(shape=points.shape[0],fill_value=(0),dtype=np.int32)
                segmentation=segmentation_ground[...,np.newaxis]
            
            # For all other instances (non-ground)
            if obj_num!=0:
                pcd=point_cloud_dic[obj_num]
                segmentation=np.full(shape=np.asarray(pcd.points).shape[0],fill_value=((self.seg_dic[class_name])),dtype=np.int32)
                segmentation = segmentation[..., np.newaxis]
                # Convert points to tensor for sampling
                torch_pts=torch.from_numpy(np.asarray(pcd.points))
                torch_pts = torch.unsqueeze(torch_pts, dim=0)
                if torch_pts.shape[1]<100:
                    point_idx = farthest_point_sampler(torch_pts, torch_pts.shape[1])
                else:  
                    point_idx = farthest_point_sampler(torch_pts, 100)

                np_arr = point_idx.cpu().detach().numpy()
                FPS_index=np.squeeze(np_arr)
             
                kdtree = o3d.geometry.KDTreeFlann(pcd)
              
                points = np.asarray(pcd.points) # Nx3 np array of points
                normals = np.asarray(pcd.normals) # Nx3 np array of normals
                
                # Ensure points and normals have the same length
                if len(normals)!= len(points):
                    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.suction_radius, max_nn=500))
                    normals = np.asarray(pcd.normals)

            
                # Placeholder lists for storing rotation and translation values
                translation_after_seal_pass=[]
                rotation_after_seal_pass=[]
                translation_after_seal_fail=[]
                rotation_after_seal_fail=[]
                translation_after_collision_pass=[]
                rotation_after_collision_pass=[]
                translation_after_collision_fail=[]
                rotation_after_collision_fail=[]
                
                # For each index in the index list

                for index in FPS_index:
                    #####################compute_darboux_frame###################################################################

                    t_ori, t_translate, R,normal_vector,normal_vector_opposite=compute_darboux_frame(points, index, kdtree, normals)
                    polygon=64
                    vertices =compute_suction_vertices(R, t_translate,polygon,args.suction_radius)
                    
                    #####################collision check###################################################################

                    for suction_prim in vertices[-1,:,:]:
                        if (check_raycast(suction_prim,normal_vector))== None:
                            collision_flag=False
                        else:
                            collision_flag=True
                            break
                            
                    ####################save collision check results##############################
                    if collision_flag== False:
                        rotation_after_collision_pass.append(R)
                        translation_after_collision_pass.append(t_ori)
                    if collision_flag== True:
                        translation_after_collision_fail.append(t_ori)
                        rotation_after_collision_fail.append(R)
                    ###############################################################################
                    

                    if collision_flag== False:
                        #####################seal evaluation###################################################################
                        collision_flag, deformation_measurement=seal_evaluation(vertices, normal_vector_opposite, stage, obj_num, polygon,collision_flag,args.suction_radius,args.deformation_thredshold)
                        ##############################################isaac_visual######################################
                        if args.headless == False and args.debug_draw == True:
                            isaac_visual(vertices, polygon,draw,deformation_measurement,collision_flag)

                            simulation_app.update()
                            for time_step in range(1):
                                the_world.step(render=True)
                            draw.clear_points()
                            draw.clear_lines()  
                        ##################################################################################################

                    ##################save seal check results############################
                    if collision_flag== False:
                        rotation_after_seal_pass.append(R)
                        translation_after_seal_pass.append(t_ori)
                    if collision_flag== True:
                        translation_after_seal_fail.append(t_ori)
                        rotation_after_seal_fail.append(R)
                    ####################################################################
                        
                if save_seal_flag == True:
                    candidates[obj_num]=dict()
                    candidates[obj_num]["segmentation_id"]= self.seg_dic[class_name]
                    candidates[obj_num]["object_name"]=class_name
                    candidates[obj_num]["rotation_after_seal_pass"]=rotation_after_seal_pass
                    candidates[obj_num]["translation_after_seal_pass"]=translation_after_seal_pass
                    candidates[obj_num]["rotation_after_seal_fail"]=rotation_after_seal_fail
                    candidates[obj_num]["translation_after_seal_fail"]=translation_after_seal_fail
                    
                    candidates[obj_num]["translation_after_collision_pass"]=translation_after_collision_pass
                    candidates[obj_num]["rotation_after_collision_pass"]=rotation_after_collision_pass
                    candidates[obj_num]["translation_after_collision_fail"]=translation_after_collision_fail
                    candidates[obj_num]["rotation_after_collision_fail"]=rotation_after_collision_fail

                    candidates[obj_num]["total_candidates"] = len(translation_after_collision_fail)+len(translation_after_collision_pass)
                    candidates[obj_num]["total_candidates_pass_collision"] = len(translation_after_collision_pass)
                    candidates[obj_num]["total_candidates_pass_seal"] = len(translation_after_seal_pass)

                    with open(self.data_root+f"/stage_{stage_ind}/"+f"stage_{stage_ind}_candidates_after_seal"+".pkl", "wb") as f:
                        pickle.dump(candidates, f)
                    f.close()

            color = np.asarray(pcd.colors)  # assuming color is another attribute of your pcd object

            points_10 = np.hstack([points, normals,color,segmentation]) #(N,10)

            point_clouds_stage=np.append(point_clouds_stage, points_10, axis=0)

        if save_pcl_flag == True:
            np.savez_compressed(self.pointcloud_root+f"/{stage_ind}.npz",point_clouds_stage)

        simulation_app.update()
 
        return None

    def __len__(self):
        return len(self.stage_root)


data_root=args.data_path
stage_root = os.listdir(data_root)
my_dataset = generate_pcl_and_seal(data_root=data_root)

for stage_ind in range(args.start_stage,args.end_stage):
    
    my_dataset.__getitem__(stage_ind,args.save_pcl_flag,args.save_pkl_flag)














