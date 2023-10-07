import open3d as o3d
import pickle
import numpy as np
import os
import numpy as np
import random
from pathlib import Path
import argparse

# ------------------ Command-line Arguments ------------------

base_dir = Path(__file__).parent
parser = argparse.ArgumentParser(description='generate_label')
parser.add_argument('--pcl_path', type=str, default=(base_dir.parent / "pointcloud_train").as_posix(), help='point cloud path')
parser.add_argument('--data_path', type=str, default=(base_dir.parent / "synthetic_data").as_posix(), help='data path')
parser.add_argument('--label_path', type=str, default=(base_dir.parent / "sim_suction_label").as_posix(), help='save label path')
parser.add_argument('--save_label_flag', type=bool, default=False, help='Save suction label as npz file')
parser.add_argument('--has_color', type=bool, default=False, help='has color channel in point cloud or not')
parser.add_argument('--start_stage', type=int,default=0, help='start stage number')
parser.add_argument('--end_stage', type=int,default=500, help='end stage number')
parser.add_argument("--suction_radius", type=float, default=1.5, help="choose suction radius")
parser.add_argument('--debug_draw', type=bool, default=False, help='debug draw') #make it False when gerneate dataset 

args = parser.parse_args()

# ------------------ Helper Functions ------------------

def find_consecutive_rows(arr, color_flag):
    """
    Identify and return groups of consecutive rows based on channel information.
    """
    consecutive_groups = []
    group_start = 0
    channel = 9 if color_flag else 6
    for index in range(1, arr.shape[0]):
        if arr[index, channel] != arr[index - 1, channel]:
            consecutive_groups.append((arr[index - 1, channel], group_start, index - 1))
            group_start = index
    consecutive_groups.append((arr[-1, channel], group_start, arr.shape[0] - 1))
    return consecutive_groups

# ------------------ Main Dataset Class ------------------

class SuctionLabelGenerator():
    """Dataset class to generate labels for suction tasks."""

    def __init__(self, pointcloud_root, data_root, label_root):
        self.pointcloud_root=pointcloud_root
        self.data_root = data_root
        self.label_root = label_root
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        scene_idx=index
        print(index)
        
        # Load point cloud data and find consecutive rows

        pointcloud_dir= self.pointcloud_root+f"/{scene_idx}.npz"
        pointcloud=np.load(pointcloud_dir,allow_pickle=True)
        
        pointcloud=pointcloud['arr_0']
        data=find_consecutive_rows(pointcloud,color_flag=args.has_color)
        
        # Load candidate data for the current scene
        candidate=self.data_root+f"/stage_{scene_idx}"+ f"/stage_{scene_idx}_seal_simulation_candidates.pkl"
        candidate_seal=self.data_root+f"/stage_{scene_idx}"+ f"/stage_{scene_idx}_candidates_after_seal.pkl"

        if os.path.isfile(candidate):
            with open(candidate, 'rb') as f:
                candidates= pickle.load(f)
        else:
            candidates=[]
        
        if os.path.isfile(candidate_seal):
            with open(candidate_seal, 'rb') as f:
                candidates_seal= pickle.load(f)
        else:
            candidates_seal=[]

        # Process data and generate labels    
        suction_label=[]
        display=[]
        segmentation_to_object = {v['segmentation_id']: k for k, v in candidates_seal.items()}
        segmentation_to_object[1558] = 0 #ground depends on the point cloud ID in pointcloud_seal_eval.py 


        if len(candidates)==0:
                suction_label=[]
                label = np.zeros(pointcloud[:,0:3].shape[0], dtype=float)
                if not os.path.exists(self.label_root):
                    os.makedirs(self.label_root, exist_ok=True)
                if args.save_label_flag:
                    np.savez_compressed(self.label_root+f"/{scene_idx}_label.npz",suction_label)         
        else:
            for group_index in range(len(data)):
                seg_id=data[group_index][0]
                object_index=segmentation_to_object[int(seg_id)]
                if object_index==0:
                    index_begin=data[group_index][1]
                    index_end=data[group_index][2]+1
                    label_ground = np.zeros(pointcloud[index_begin:index_end,0:3].shape[0], dtype=float)
                    suction_label.append(list(label_ground))
                    point_cloud_dic={}
                    point_cloud_dic = o3d.geometry.PointCloud()
                    point_cloud_dic.points = o3d.utility.Vector3dVector(pointcloud[index_begin:index_end,0:3])
                    point_cloud_dic.normals = o3d.utility.Vector3dVector(pointcloud[index_begin:index_end,3:6])
                    if args.has_color:
                        point_cloud_dic.colors = o3d.utility.Vector3dVector(pointcloud[index_begin:index_end,6:9])
                    else:
                        point_cloud_dic.paint_uniform_color([0, 0, 1])
                    display.append(point_cloud_dic)
                else:
                    point_cloud_dic={}
                    point_cloud_dic = o3d.geometry.PointCloud()
                    index_begin=data[group_index][1]
                    index_end=data[group_index][2]+1
                    point_cloud_dic.points = o3d.utility.Vector3dVector(pointcloud[index_begin:index_end,0:3])
                    point_cloud_dic.normals = o3d.utility.Vector3dVector(pointcloud[index_begin:index_end,3:6])
                    if args.has_color:
                        point_cloud_dic.colors = o3d.utility.Vector3dVector(pointcloud[index_begin:index_end,6:9])
                    else:
                        r = random.random()
                        g = random.random()
                        b = random.random()
                        point_cloud_dic.paint_uniform_color([r, g, b])

                    if object_index in candidates.keys():
                        translation_candidates=candidates[object_index]["translation_after_exp_success"]
                        kdtree = o3d.geometry.KDTreeFlann(point_cloud_dic)
                        label = np.zeros(pointcloud[index_begin:index_end,0:3].shape[0], dtype=float)
                        for suction_number in range(len(translation_candidates)):

                            t=translation_candidates[suction_number]

                            (_, indices, _) = kdtree.search_radius_vector_3d(t,args.suction_radius)
                                                
                            pcl=np.array(point_cloud_dic.points)

                            matching_rows = np.where(np.all(pcl==t, axis=1))[0]
                            #print(matching_rows)
                            if matching_rows.size > 0:
                                index_t = matching_rows[0]
                            else:
                                # Handle the case where no matching rows are found
                                # For example, you could set index_t to a default value or raise a custom error
                                index_t = None  # or some other default value
                                # or
                                # raise ValueError(f"No matching rows found for value {t}")

                            indices = np.concatenate((indices, np.array([index_t])))
                            indices = np.array([idx for idx in indices if idx is not None], dtype=int)

                            np.asarray(point_cloud_dic.colors)[indices[1:], :] = [0, 1, 0]
                            
                            for inx in indices:
                                label[inx]=1

                        suction_label.append(list(label))
                    else:
                        
                        label = np.zeros(pointcloud[index_begin:index_end,0:3].shape[0], dtype=float)
                        suction_label.append(list(label))
                    display.append(point_cloud_dic)

            suction_label=list(np.concatenate(suction_label).flat)

            if args.save_label_flag:
                np.savez_compressed(self.label_root+f"/{scene_idx}_label.npz",suction_label)
            if args.debug_draw:
                o3d.visualization.draw_geometries_with_custom_animation(display,width=720,height=720)


        return 
    

if __name__ == "__main__":


    data_root=args.data_path
    label_root=args.label_path
    pointcloud_root=args.pcl_path

    my_dataset = SuctionLabelGenerator(pointcloud_root=pointcloud_root,data_root=data_root,label_root=label_root)

    for stage_ind in range(args.start_stage,args.end_stage):
        my_dataset.__getitem__(stage_ind)



