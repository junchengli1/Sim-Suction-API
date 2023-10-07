import numpy as np
import open3d as o3d
import argparse
from pathlib import Path

def visualize_point_cloud(data_path, stage, frame):
    npz_file = Path(data_path) / f"{stage}_{frame}.npz"

    if not npz_file.exists():
        print(f"File {npz_file} does not exist!")
        return

    data = np.load(npz_file, allow_pickle=True)
    points_10 = data['arr_0']
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_10[:,:3])
    point_cloud.normals = o3d.utility.Vector3dVector(points_10[:,3:6])
    point_cloud.colors = o3d.utility.Vector3dVector(points_10[:,6:9])

    # Visualize
    o3d.visualization.draw_geometries([point_cloud])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize point cloud from a .npz file.")
    parser.add_argument('--dataset_type', type=str, default='test_novel', choices=['test_similar', 'test_novel'], help='Choose dataset type: similar or novel')
    parser.add_argument('--stage', type=int, default=0,help="Stage number to visualize.")
    parser.add_argument('--frame', type=int, default=0, help="Frame number to visualize.")
    
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    if args.dataset_type == 'test_similar':
        data_path = (base_dir.parent / "test_similar_pointcloud").as_posix()
    else:
        data_path = (base_dir.parent / "test_novel_pointcloud").as_posix()
    
    visualize_point_cloud(data_path, args.stage, args.frame)
