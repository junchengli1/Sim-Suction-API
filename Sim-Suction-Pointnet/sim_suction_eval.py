# Omniverse Libraries
from omni.isaac.kit import SimulationApp
from pathlib import Path
import argparse
import sys

base_dir = Path(__file__).parent
parser = argparse.ArgumentParser(description='Sim-Suction create_point_cloud_and_seal_evaluation')
parser.add_argument('--headless', type=bool, default=True, help='headless')#make it True when gerneate dataset 
parser.add_argument('--debug_draw', type=bool, default=False, help='debug draw')#make it False when gerneate dataset 
parser.add_argument('--pcl_path', type=str, default=(base_dir.parent / "pointcloud_train").as_posix(), help='point cloud path')
parser.add_argument('--data_path', type=str, default=(base_dir.parent / "test_similar").as_posix(), help='data path')
parser.add_argument('--instanceable_flag', type=bool, default=False, help='use textureless instanceable usd to increase simulation speed')
parser.add_argument('--seg_dic_path', type=str, default=(base_dir.parent / "seg_dic.pkl").as_posix(), help='seg_dic path')
parser.add_argument('--save_pkl_flag', type=bool, default=True, help='Save seal evaluation as pkl file')
parser.add_argument('--suction_radius', type=float, default=1.0, help='suction radius 1.5cm')
parser.add_argument('--deformation_thredshold', type=float, default=0.25, help='15 percent deformation')
parser.add_argument('--start_stage', type=int,default=0, help='start stage number')
parser.add_argument('--end_stage', type=int,default=500, help='end stage number')
parser.add_argument('--save_result_path', type=str, default= (base_dir.parent / "test_similar_inference_result").as_posix(), choices=["../test_similar_inference_result","../test_novel_inference_result"], help='save result path')
parser.add_argument('--save_eval_path', type=str, default= (base_dir.parent / "test_similar_eval_result").as_posix(), choices=["../test_similar_inference_result","../test_novel_inference_result"], help='save result path')


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
sim_utils_dir = base_dir.parent / "isaac_sim_gen"
sys.path.insert(0, sim_utils_dir.as_posix())
from simulation_utils import *  # This imports all functions from the utility file


def calculate_precision(result_dict, top_indices):
    """
    Calculate the precision based on the result_dict and the top5_indices.

    Args:
    - result_dict (dict): Dictionary with indices as keys and boolean values indicating True Positive.
    - top5_indices (list): List of indices in the top 5%.

    Returns:
    - float: Precision value.
    """
    
    # Count of True Positives
    TP = sum([1 for index in top_indices if not result_dict.get(index, True)])

    
    # Total number of predictions 
    total_predictions = len(top_indices)
    
    # Precision calculation
    precision = TP / total_predictions if total_predictions != 0 else 0
    
    return TP,total_predictions,precision



class eval_one_frame():
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
    
    def __getitem__(self, stage_ind,save_seal_flag):
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

     
    
        # For each unique class, combine point clouds from all cameras, and perform raycasting and collision detection
       
        draw = _debug_draw.acquire_debug_draw_interface()

        draw.clear_points()
        draw.clear_lines()
        
        # Create a map for object number to label
       
        # Initial lists and dictionaries
        display=[]
       
        if not os.path.exists(args.save_eval_path):
            os.makedirs(args.save_eval_path)
        
        for frame in range(100):
            candidates={}
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

            inference_result=args.save_result_path+f"/stage_{stage_ind}_{frame}_results.pkl"
            
            if not os.path.exists(inference_result):
                continue

            with open(inference_result, 'rb') as f:
                inference_result= pickle.load(f)
            
            top_10_result=inference_result["top_10%"]
            ind_list=len(inference_result["top_10%"]["t_ori"])
            
            dict={}

            for index in range(ind_list):
                #####################compute_darboux_frame###################################################################
                t_ori=top_10_result["t_ori"][index]
                t_translate=top_10_result["t_translate"][index]
                R=top_10_result["R"][index]
                normal_vector=top_10_result["normal_vector"][index]
                normal_vector_opposite=top_10_result["normal_vector_opposite"][index]

                polygon=16
                vertices =compute_suction_vertices(R, t_translate,polygon,1)
                
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
                    collision_flag, deformation_measurement=seal_evaluation(vertices, normal_vector_opposite, stage, None, polygon,collision_flag,1,0.35)
                    ##############################################isaac_visual######################################
                    if args.headless == False and args.debug_draw == True:
                        isaac_visual(vertices, polygon,draw,deformation_measurement,collision_flag)

                        simulation_app.update()
                        for time_step in range(1):
                            the_world.step(render=True)
                        draw.clear_points()
                        draw.clear_lines()  
                    ##################################################################################################
                    simulation_app.update()
                ##################save seal check results############################
                if collision_flag== False:
                    dict[index]=collision_flag
                    rotation_after_seal_pass.append(R)
                    translation_after_seal_pass.append(t_ori)
                if collision_flag== True:
                    translation_after_seal_fail.append(t_ori)
                    rotation_after_seal_fail.append(R)
                ####################################################################
                



            

            TP_1,total_predictions_1,AP_1=calculate_precision (dict,inference_result["top_1"]["relative_indices"])
            TP_1per,total_predictions_1per,AP_1per=calculate_precision (dict,inference_result["top_1%"]["relative_indices"])
            TP_5per,total_predictions_5per,AP_5per=calculate_precision (dict,inference_result["top_5%"]["relative_indices"])

            candidates["top_10%"]={}
            candidates["top_10%"]["total_candidates"] = len(translation_after_collision_fail)+len(translation_after_collision_pass)
            candidates["top_10%"]["TP"] = len(translation_after_seal_pass)
            candidates["top_10%"]["AP"] = len(translation_after_seal_pass)/(len(translation_after_collision_fail)+len(translation_after_collision_pass))

            candidates["top_1"]={}
            candidates["top_1"]["total_candidates"] = total_predictions_1
            candidates["top_1"]["TP"] = TP_1
            candidates["top_1"]["AP"] = AP_1
            
            candidates["top_1%"]={}
            candidates["top_1%"]["total_candidates"] = total_predictions_1per
            candidates["top_1%"]["TP"] = TP_1per
            candidates["top_1%"]["AP"] = AP_1per
            
            candidates["top_5%"]={}
            candidates["top_5%"]["total_candidates"] = total_predictions_5per
            candidates["top_5%"]["TP"] = TP_5per
            candidates["top_5%"]["AP"] = AP_5per
            
            print(f'---- Evaluate stage: {stage_ind}, frame: {frame} ----')


            print(candidates)

            if save_seal_flag == True:

                with open(args.save_eval_path+f"/stage_{stage_ind}_{frame}_results.pkl", "wb") as f:
                        pickle.dump(candidates, f)
                f.close()


            simulation_app.update()
    
        return None

    def __len__(self):
        return len(self.stage_root)


data_root=args.data_path
stage_root = os.listdir(data_root)
my_dataset = eval_one_frame(data_root=data_root)

for stage_ind in range(args.start_stage,args.end_stage):
    
        my_dataset.__getitem__(stage_ind,args.save_pkl_flag)














