from torch.utils.data import Dataset
import torch
import os
import numpy as np

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def rotate_point_cloud_with_normal(xyz_normal):
    ''' Randomly rotate XYZ, normal point cloud.
        Input:
            xyz_normal: N,6, first three channels are XYZ, last 3 all normal
        Output:
            N,6, rotated XYZ, normal point cloud
    '''
    rotation_angle = np.random.uniform() * np.pi/4
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    
    shape_pc = xyz_normal[:, 0:3]
    shape_normal = xyz_normal[:, 3:6]
    
    xyz_normal[:, 0:3] = np.dot(shape_pc, rotation_matrix)
    xyz_normal[:, 3:6] = np.dot(shape_normal, rotation_matrix)

    return xyz_normal

def random_scale_point_cloud(data, scale_low=0.9, scale_high=1.2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    N, C = data.shape
    scales = np.random.uniform(scale_low, scale_high)
    #for batch_index in range(n):
    data[:,:] *= scales
    return data


class SimSuctionDataset(Dataset):
    def __init__(self, data_root, suction_label_root):
        self.data_root = data_root
        self.label_root = suction_label_root
        stage_root = os.listdir(data_root)
        self.room_points, self.room_labels,= {},{}
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        sample_rate=4
        block_size=60
        self.block_size=block_size
        self.num_point=20000
        num_point=self.num_point
        exclusion_list = []
        points=np.load(self.data_root+f"/{0}.npz",allow_pickle=True)['arr_0']
        for scene in range(0,len(stage_root)):
            if scene in exclusion_list:
                continue   
            num_point_all.append(points.size)
 
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        filtered_stage_root = [i for i in range(len(stage_root)) if i not in exclusion_list]
        room_idxs = [index for index in filtered_stage_root for _ in range(int(round(sample_prob[filtered_stage_root.index(index)] * num_iter)))]
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), "train"))
        print(self.room_idxs)

    def __len__(self):
        return len(self.room_idxs)

    def __getitem__(self, index):
        scene_idx=index
        #print(scene_idx)

        room_idx = self.room_idxs[index]
        points=np.load(self.data_root+f"/{room_idx}.npz",allow_pickle=True)['arr_0']

        if not os.path.isfile(self.label_root+f"/{room_idx}_label.npz"):
            labels=np.zeros((len(points),), dtype=int)
        else:
            labels=np.load(self.label_root+f"/{room_idx}_label.npz",allow_pickle=True)['arr_0']
        #print(labels)
        N_points = points.shape[0]
        if labels.size==0:
            labels=np.zeros((len(points),), dtype=int)

        center = points[np.random.choice(N_points)][:3]
        block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
        block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
        point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
      
        
        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)

        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points=selected_points[:,0:6]

        current_points[:, 0:3]=random_scale_point_cloud(selected_points[:, 0:3])
        current_points[:, 0:3] = pc_normalize(current_points[:, 0:3])

        current_labels = labels[selected_point_idxs]
        
        current_labels = current_labels.astype(np.int64)

        points_w_feature = np.zeros((self.num_point, 6))
        points_w_feature[:, :6] = current_points
      
        points_w_feature =rotate_point_cloud_with_normal(points_w_feature)
        
        
        pushness_score = current_labels[:, np.newaxis]  # reshapes (N,) to (N, 1)
        return torch.from_numpy(points_w_feature), torch.from_numpy(pushness_score)




if __name__ == "__main__":
    data_root="../pointcloud_train"
    suction_label_root="../sim_suction_label"
        
    my_dataset = SimSuctionDataset(data_root,suction_label_root)
    
    for i in range(0,500):
        
        my_dataset.__getitem__(i)











