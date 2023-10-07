from omni.isaac.kit import SimulationApp
from pathlib import Path
import argparse

base_dir = Path(__file__).parent

parser = argparse.ArgumentParser(description='generate single shot pointcloud for test similar or test_novel')
parser.add_argument('--headless', type=bool, default=False, help='headless')
parser.add_argument('--save_pcl_flag', type=bool, default=True, help='Save single shot point cloud as npz file')
parser.add_argument('--dataset_type', type=str, default='test_novel', choices=['test_similar', 'test_novel'], help='Choose dataset type: similar or novel')
parser.add_argument('--frame_per_stage', type=int, default=100, help='how many frames per stage')
parser.add_argument('--start_stage', type=int, default=0, help='start stage number')
parser.add_argument('--end_stage', type=int, default=100, help='end stage number')
parser.add_argument('--visualization_flag', type=bool, default=False, help='visualize the point cloud')

args = parser.parse_args()

if args.dataset_type == 'test_similar':
    args.data_path = (base_dir.parent / "test_similar").as_posix()
    args.seg_dic_path = (base_dir.parent / "seg_dic.pkl").as_posix()
    args.save_pcl_path = (base_dir.parent / "test_similar_pointcloud").as_posix()
else:
    args.data_path = (base_dir.parent / "test_novel").as_posix()
    args.seg_dic_path = (base_dir.parent / "novel_seg_dic.pkl").as_posix()
    args.save_pcl_path = (base_dir.parent / "test_novel_pointcloud").as_posix()

simulation_app = SimulationApp({"headless": args.headless})

from omni.isaac.core import World
from pxr import UsdGeom,Usd
from omni.isaac.core.utils.prims import create_prim
import numpy as np
import os
import glob
import pickle
import random
from scipy.spatial.transform import Rotation as R
import os
import pickle
import math as m
import open3d as o3d
import omni.physx
from pxr import UsdGeom
import math
import omni.replicator.core as rep
from numpy import save

import asyncio
from simulation_utils import *  # This imports all functions from the utility file
from omni.physx.scripts import utils

class single_shot_rep():
    def __init__(self, data_root):
        self.data_root = data_root
        self.stage_root = os.listdir(data_root)
        self.d=self.new_positions_spherical_coordinates()

    async def load_stage(self, path):
        await omni.usd.get_context().open_stage_async(path)

    def new_positions_spherical_coordinates(self):
        radius = np.random.uniform(50.0,400.0, (10200,1)) 
        theta = np.random.uniform(0,1,(10200,1))*math.pi
        phi = np.arccos(1-2*np.random.uniform(0,1,(10200,1)))
        x = radius * np.sin( phi ) * np.cos( theta )
        y = radius * np.sin( phi ) * np.sin( theta )
        z = radius * np.cos( phi )
        ar = np.array(y)
        y=ar+50
        return (x,y,z)

    def __getitem__(self, index):
     
        with open(args.seg_dic_path, "rb") as f:
                self.seg_dic=pickle.load(f)
        f.close()

        
        stage_usd= self.data_root+f"/stage_{index}"+ f"/stage_{index}.usd"

        setup_task = asyncio.ensure_future(self.load_stage(stage_usd))
        
        while not setup_task.done():
            simulation_app.update()
   
        the_world = World(stage_units_in_meters = 0.01)
        stage = omni.usd.get_context().get_stage()

        RANDOM_TRANSLATION_X = (-450, 450)
        RANDOM_TRANSLATION_Y= (-450, 450)
        RANDOM_TRANSLATION_Z = (800.0, 3200)
        x = random.uniform(*RANDOM_TRANSLATION_X)
        y = random.uniform(*RANDOM_TRANSLATION_Y)
        z = random.uniform(*RANDOM_TRANSLATION_Z)


        intensity_light=(3000, 20000)
        INTENSITY = random.uniform(*intensity_light)
        
        radius_light=(50, 200)
        RADIUS_LIGHT = random.uniform(*radius_light)


        LIGHTING_LIST=[[255/255, 147/255, 41/255],[255/255, 197/255, 143/255],[255/255, 214/255, 170/255],[255/255, 241/255, 224/255],[255/255, 250/255, 244/255],[255/255, 255/255, 251/255],[255/255, 255/255, 255/255]]


        r=random.choice(LIGHTING_LIST)
    
        create_prim(
            "/World/Light1",
            "SphereLight",
            position=np.array([x, y, z]),
            attributes={"radius": RADIUS_LIGHT, "intensity": INTENSITY, "color": (r[0],r[1], r[2])},
            )
            
        
        the_world.play()
        for i in range(150):
                the_world.step(render=True)    
        
        curr_prim = stage.GetPrimAtPath("/World/objects")
        for prim in Usd.PrimRange(curr_prim):
            if prim.IsA(UsdGeom.Xform):
                utils.removeRigidBody(prim)
            elif prim.IsA(UsdGeom.Mesh):
                utils.setCollider(prim, approximationShape="none")

        stage_folder=self.data_root+f"/stage_{index}"

        with rep.new_layer():
            
            W, H = (1280, 720)
      
            camera = rep.create.camera(position=(100, 100, 100), look_at=(0,0,0))
        
            render_product = rep.create.render_product(camera, (W, H))
        
            pcd_annotators=[]
            sem_annots=[]

            pointcloud_anno = rep.annotators.get("pointcloud")
            semantic_segmentation_anno = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
            pointcloud_anno.attach([render_product])
            semantic_segmentation_anno.attach([render_product])
            pcd_annotators.append(pointcloud_anno)
            sem_annots.append(semantic_segmentation_anno)

            for frame in range(args.frame_per_stage):
                camera_pose_x=self.d[2][frame]
                camera_pose_y=self.d[0][frame]
                camera_pose_z=self.d[1][frame]
                with camera:
                    rep.modify.pose(position=(camera_pose_x, camera_pose_y, camera_pose_z), look_at=(0,0,0))
                print(f"stage_frame:{index}_{frame}")
                rep.orchestrator.step()

                all_unique_classes, label_to_object_number_map=get_semantic_info(pcd_annotators,sem_annots)

                point_cloud_dic,all_instance_seg_ids=get_merge_pointcloud(all_unique_classes,pcd_annotators,label_to_object_number_map,stage_folder,False)

                object_number_to_label_map = {v: k for k, v in label_to_object_number_map.items()}

                point_clouds_stage=np.empty((0,10), float)

                display=[]
                for obj_num in all_instance_seg_ids:
                    class_name=object_number_to_label_map[obj_num]
                    print(object_number_to_label_map[obj_num])
                    # If current instance is ground
                    if obj_num==0:
                        pcd=point_cloud_dic[obj_num]
                        display.append(pcd)

                        points=np.asarray(pcd.points)
                        normals=np.asarray(pcd.normals)
                        colors=np.asarray(pcd.colors)
                        asset=stage.GetPrimAtPath("/World/objects")
                        bound = UsdGeom.Mesh(asset).ComputeWorldBound(0.0, "default")
                        box_min_y = bound.GetBox().GetMin()[1]
                        box_max_y = bound.GetBox().GetMax()[1]
                        box_min_x = bound.GetBox().GetMin()[0]
                        box_max_x = bound.GetBox().GetMax()[0]
                        
                        block_min =  [box_min_x, box_min_y, 0]
                        block_max =  [box_max_x, box_max_y, 0]
                        point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
                        

                        points=points[point_idxs]
                        normals=normals[point_idxs]
                        colors=colors[point_idxs]

                        segmentation_ground=np.full(shape=points.shape[0],fill_value=(0),dtype=np.int32)
                        segmentation=segmentation_ground[...,np.newaxis]

                        
                    # For all other instances (non-ground)
                    if obj_num!=0:
                        pcd=point_cloud_dic[obj_num]
                        display.append(pcd)
                        segmentation=np.full(shape=np.asarray(pcd.points).shape[0],fill_value=((self.seg_dic[class_name])),dtype=np.int32)
                        segmentation = segmentation[..., np.newaxis]
                        points = np.asarray(pcd.points) # Nx3 np array of points
                        normals = np.asarray(pcd.normals) # Nx3 np array of normals
                        colors=np.asarray(pcd.colors)


                    points_10 = np.hstack([points, normals,colors,segmentation]) #(N,10)

                    point_clouds_stage=np.append(point_clouds_stage, points_10, axis=0)
                
                if args.visualization_flag:
                    o3d.visualization.draw_geometries_with_custom_animation(display,width=720,height=720)

                if args.save_pcl_flag:
                    np.savez_compressed(args.save_pcl_path+f"/{index}_{frame}.npz",points_10)

                simulation_app.update()
  
        return None

    def __len__(self):
        return len(self.stage_root)

data_root=args.data_path

if not os.path.exists(args.save_pcl_path):
    os.makedirs(args.save_pcl_path, exist_ok=True)

my_dataset = single_shot_rep(data_root=data_root)

for stage in range(args.start_stage,args.end_stage):
    my_dataset.__getitem__(stage)
    













