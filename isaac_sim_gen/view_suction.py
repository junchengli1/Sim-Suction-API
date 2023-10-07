#__author__ = 'Juncheng_Li'
#__contact__ = 'li3670@purdue.edu'

import open3d as o3d
import glob
import pickle
import numpy as np
import sys
import argparse
from pathlib import Path


"""Parse command-line arguments."""

parser = argparse.ArgumentParser()

base_dir = Path(__file__).parent
parser = argparse.ArgumentParser(description='Dataset_generation')

parser.add_argument('--data_path', type=str, default=(base_dir.parent / "synthetic_data").as_posix(), help='data path')
parser.add_argument("--stage_ID", type=int, default=0, help='stage ID number [default: 0]')
parser.add_argument("--mode", default="simulation", choices=["collision","seal","simulation"],help="choose the type of check to visualize")
parser.add_argument("--ground", default=False, choices=[True,False],help="whether to include ground plane")
parser.add_argument("--suction_radius", type=float, default=1.5, help="choose suction radius 1.5cm")

args = parser.parse_args()

DATA_ROOT = args.data_path
STAGE_ID=args.stage_ID
MODE=args.mode
GROUND_PLANE=args.ground
RADIUS=args.suction_radius



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

def draw(DATA_ROOT,STAGE_ID,MODE,GROUND_PLANE,RADIUS):
    display=[]
    pcd_list=glob.glob(DATA_ROOT+ f"/stage_{STAGE_ID}"+"/**/*.pcd", recursive=True)
    if GROUND_PLANE==False:
       pcd_list.remove(DATA_ROOT+ f"/stage_{STAGE_ID}"+"/ground.pcd")
    if RADIUS==1.5:
        candidate_seal_path=DATA_ROOT+f"/stage_{STAGE_ID}"+ f"/stage_{STAGE_ID}_candidates_after_seal.pkl"
        candidate_simulation_path=DATA_ROOT+f"/stage_{STAGE_ID}"+ f"/stage_{STAGE_ID}_seal_simulation_candidates.pkl"

    with open(candidate_seal_path, 'rb') as f:
        candidate_seal= pickle.load(f)
 

    if MODE=="collision":
        for object_index in candidate_seal.keys():

            translation_candidates=candidate_seal[object_index]["translation_after_collision_pass"]
            rotation_candidates=candidate_seal[object_index]["rotation_after_collision_pass"]
            translation_candidates_bad=candidate_seal[object_index]["translation_after_collision_fail"]
            rotation_candidates_bad=candidate_seal[object_index]["rotation_after_collision_fail"]

            for i in range(len(translation_candidates_bad)):
                t_bad=translation_candidates_bad[i]
                R_bad=rotation_candidates_bad[i]
                collision=True
                mesh=create_mesh_cylinder_detection(R_bad,t_bad,collision)
                display.append(mesh)
            for i in range(len(translation_candidates)):
                t=translation_candidates[i]
                R=rotation_candidates[i]
                collision=False

                mesh=create_mesh_cylinder_detection(R,t,collision)
                display.append(mesh)
        

    if MODE=="seal":
        for object_index in candidate_seal.keys():
            
            translation_candidates=candidate_seal[object_index]["translation_after_seal_pass"]
            rotation_candidates=candidate_seal[object_index]["rotation_after_seal_pass"]
            translation_candidates_bad=candidate_seal[object_index]["translation_after_seal_fail"]
            rotation_candidates_bad=candidate_seal[object_index]["rotation_after_seal_fail"]
                
            for i in range(len(translation_candidates_bad)):
                t_bad=translation_candidates_bad[i]
                R_bad=rotation_candidates_bad[i]
                collision=True
                mesh=create_mesh_cylinder_detection(R_bad,t_bad,collision)
                display.append(mesh)


            for i in range(len(translation_candidates)):
                t=translation_candidates[i]
                R=rotation_candidates[i]
                
                collision=False
                mesh=create_mesh_cylinder_detection(R,t,collision)
                display.append(mesh)
            

    if MODE=="simulation":
        with open(candidate_simulation_path, 'rb') as f:
            candidate_simulation= pickle.load(f)
        for object_index in candidate_seal.keys():
            
            translation_candidates=candidate_seal[object_index]["translation_after_seal_pass"]
            rotation_candidates=candidate_seal[object_index]["rotation_after_seal_pass"]
            translation_candidates_bad=candidate_seal[object_index]["translation_after_seal_fail"]
            rotation_candidates_bad=candidate_seal[object_index]["rotation_after_seal_fail"]
                
            for i in range(len(translation_candidates_bad)):
                t_bad=translation_candidates_bad[i]
                R_bad=rotation_candidates_bad[i]
                collision=True
                mesh=create_mesh_cylinder_detection(R_bad,t_bad,collision)
                display.append(mesh)


        for object_index in candidate_simulation.keys():
        
            if 'rotation_after_exp_fail' in candidate_simulation[object_index].keys():
                translation_candidates=candidate_simulation[object_index]["translation_after_exp_success"]
                rotation_candidates=candidate_simulation[object_index]["rotation_after_exp_success"]
                translation_candidates_bad=candidate_simulation[object_index]["translation_after_exp_fail"]
                rotation_candidates_bad=candidate_simulation[object_index]["rotation_after_exp_fail"]
                for i in range(len(translation_candidates)):
                    t=translation_candidates[i]
                    R=rotation_candidates[i]
                    collision=False
                    mesh=create_mesh_cylinder_detection(R,t,collision)
                    display.append(mesh)
                for i in range(len(translation_candidates_bad)):
                    t_bad=translation_candidates_bad[i]
                    R_bad=rotation_candidates_bad[i]
                    collision=True
                    mesh=create_mesh_cylinder_detection(R_bad,t_bad,collision)
                    display.append(mesh)
    
    for i in pcd_list:

        pcd = o3d.io.read_point_cloud(i)
        display.append(pcd)

    o3d.visualization.draw_geometries_with_custom_animation(display,width=720,height=720)

if __name__ == "__main__":
    draw(DATA_ROOT,STAGE_ID,MODE,GROUND_PLANE,RADIUS)
