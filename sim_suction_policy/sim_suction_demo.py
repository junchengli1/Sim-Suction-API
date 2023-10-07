import argparse
import os
import copy
import numpy as np
import torch
import sys
import matplotlib
import matplotlib.pyplot as plt
import cv2
import open3d as o3d

from sim_suction_policy_utils import *
from sim_suction_policy_model import sim_suction_policy_model
#matplotlib.use('Agg')
# Get the directory of the current script
current_directory = os.path.abspath(os.path.dirname(__file__))

# Grounding DINO
from GroundingDINO.groundingdino.util.utils import  get_phrases_from_posmap

#################################################################################################

parser = argparse.ArgumentParser("Sim-Suction-Policy Demo", add_help=True)
#use all objects prompt if you want to pick up all objects and select the best to grasp#
#use one object prompt if you want to pick up one object#
parser.add_argument("--text_prompt", type=str, default="all objects", help="text prompt picking up *****")
parser.add_argument("--box_threshold", type=float, default=0.27, help="box threshold")
parser.add_argument("--text_threshold", type=float, default=0.3, help="text threshold")
parser.add_argument("--device", type=str, default="cuda", help="device")
parser.add_argument("--demo_path", type=str, default=current_directory+"/demo", help="device")
parser.add_argument("--demo_number", type=str, default="demo")
#if suction confident score is lower than this threshold, the suction pose will not be considered as good pose#
parser.add_argument("--suction_confident_threshold", type=float, default=0.2, help="confident threshold to output suction poses")

args = parser.parse_args()

text_prompt = args.text_prompt
box_threshold = args.box_threshold
text_threshold = args.text_threshold
device = args.device
demo_path=args.demo_path
demo_number=args.demo_number
suction_confident_threshold=args.suction_confident_threshold

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases
 
def get_mask_dino_sam(predictor,model,image_pil,image):

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )


    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )
    
    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)

    plt.axis('off')
    #plt.show()

    return masks

def remove_points_from_pointcloud(base_pcd, points_to_remove):
    """
    Removes points from base_pcd that are closest to points_to_remove.
    
    Args:
    - base_pcd (o3d.geometry.PointCloud): The point cloud from which points are to be removed.
    - points_to_remove (o3d.geometry.PointCloud): The point cloud containing points to be removed from base_pcd.
    
    Returns:
    - o3d.geometry.PointCloud: The modified point cloud.
    """
    
    # Build a KDTree for the base point cloud
    kdtree = o3d.geometry.KDTreeFlann(base_pcd)
    
    # For each point in points_to_remove, find the closest point in base_pcd
    indices_to_remove = []
    for point in np.asarray(points_to_remove.points):
        _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
        indices_to_remove.append(idx[0])
    
    # Remove the identified indices from base_pcd
    base_points = np.asarray(base_pcd.points)
    base_colors = np.asarray(base_pcd.colors)
    mask = np.ones(base_points.shape[0], dtype=bool)
    mask[indices_to_remove] = False
    base_pcd.points = o3d.utility.Vector3dVector(base_points[mask])
    if base_colors.size > 0:
        base_pcd.colors = o3d.utility.Vector3dVector(base_colors[mask])
    
    return base_pcd

def preprocess_data(pcd, segmentation):
    """Preprocess point cloud and segmentation data."""
    points_read = np.zeros((len(pcd.points), 6))
    points_read[:, 0:3] = np.array(pcd.points)
    points_read[:, 3:6] = np.array(pcd.normals)
    point_idxs = np.where(segmentation != 1)[0]
    points_ori = copy.deepcopy(points_read[point_idxs, 0:3])
    points_filter = points_read[point_idxs]
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, 1)
    if len(points_filter) < 5120:
            # Calculate the number of points needed
            num_points_needed = 5120 - len(points_filter)           
            zero_padding = np.zeros((num_points_needed, points_filter.shape[1]))
            points_filter = np.vstack((points_filter, zero_padding))

    return points_filter, points_ori, point_idxs,voxel_grid

def evaluate_suction_model(points,point_idxs, score_model):
    points[:, 0:3] = pc_normalize(points[:, 0:3])

    points=torch.from_numpy(points).float(). unsqueeze(0)
    
    pc = points.cuda()

    _,output_score = score_model(pc, None)
    camera_resolution = (720, 1280)

    camera_res_idx = np.unravel_index(point_idxs, camera_resolution)
    camera_res_idx = np.array(camera_res_idx).T
    score_map = np.zeros(camera_resolution)
    for idx, score in zip(camera_res_idx, output_score):
        score_map[idx[0], idx[1]] = score
    return score_map,camera_res_idx

def create_masked_score_map(instance_mask):
    instance_mask = instance_mask.cpu().numpy()
    # Convert the instance_mask to binary format
    binary_mask = (instance_mask > 0).astype(np.float32)
    binary_mask=np.squeeze(binary_mask)
    #print(binary_mask.shape)
    centroid = calculate_centroid(binary_mask)
    dimensions = binary_mask.shape
    decay_factor=0.1
    heatmap = create_heatmap_centroid(dimensions, centroid, decay_factor)       
    masked_score_map = heatmap*score_map* binary_mask
    return masked_score_map

def get_smoothed_masked_score_map(masked_score_map):
    """Smooth the masked score map (if required)."""
    # smoothed = gaussian_filter(masked_score_map, sigma) # Uncomment if you want to use Gaussian filter
    return np.squeeze(masked_score_map)

def get_segmented_point_cloud_and_scores(points_ori, camera_res_idx, smoothed_masked_score_map, image_pil):
    """Retrieve segmented point cloud and corresponding scores."""
    smoothed_predicted_scores_instance = smoothed_masked_score_map[camera_res_idx[:, 0], camera_res_idx[:, 1]]
    rgb_colors = np.array(image_pil)[camera_res_idx[:, 0], camera_res_idx[:, 1]]
    non_zero_indices = np.nonzero(smoothed_predicted_scores_instance)[0]
    segmented_point_cloud = points_ori[non_zero_indices]
    return segmented_point_cloud, smoothed_predicted_scores_instance, smoothed_predicted_scores_instance[non_zero_indices], rgb_colors[non_zero_indices]

def visualize_segmented_points(segmented_point_cloud, normalized_scores, rgb_colors):
    """Visualize segmented points using a colormap."""
    cmap = matplotlib.cm.get_cmap('plasma')
    score_colors = cmap(normalized_scores)
    alpha = 0.7
    blended_colors = alpha * score_colors[:, :3] + (1 - alpha) * rgb_colors / 255.0
    segmented_pcd = o3d.geometry.PointCloud()
    segmented_pcd.points = o3d.utility.Vector3dVector(np.array(segmented_point_cloud))
    segmented_pcd.colors = o3d.utility.Vector3dVector(blended_colors[:, :3])
    return segmented_pcd

def calculate_collision_free_suction_candidates(suction_confident_threshold,points_ori, kdtree, normals, smoothed_predicted_scores_instance, sort_idx, voxel_grid, display):
    for idx in sort_idx:
        t_ori, t_translate, Rotation_mat, normal_vector, _ = compute_darboux_frame(
            points_ori, idx, kdtree, normals)
        collision = check_collision_with_suction_gripper_using_voxelgrid(
            voxel_grid, t_translate, normal_vector)
        # If no collision, append to lists and break out of the loop to consider this as top-1 point
        if not collision:
            if smoothed_predicted_scores_instance[idx]>suction_confident_threshold:
                print("confident score:", smoothed_predicted_scores_instance[idx])
                confident=smoothed_predicted_scores_instance[idx]
                mesh = create_mesh_cylinder_detection_based_on_alpha(Rotation_mat, t_ori, confident,radius=1.5)
                display.append(mesh)
            break
    return display

def process_masks(suction_confident_threshold,masks, normals, camera_res_idx, points_ori, kdtree, voxel_grid, image_pil, global_min_score, global_max_score):
    """Process masks to obtain visualizations and top suction points."""
    display = []
    combined_pcd = o3d.geometry.PointCloud()
    for instance_mask in masks:
        masked_score_map=create_masked_score_map(instance_mask)
        
        smoothed_masked_score_map = get_smoothed_masked_score_map(masked_score_map)

        segmented_point_cloud, smoothed_predicted_scores_instance,segmented_scores, rgb_colors = get_segmented_point_cloud_and_scores(
            points_ori, camera_res_idx, smoothed_masked_score_map, image_pil)

        normalized_scores = (segmented_scores - global_min_score) / (global_max_score - global_min_score)
        segmented_pcd = visualize_segmented_points(segmented_point_cloud, normalized_scores, rgb_colors)
        combined_pcd += segmented_pcd

        sort_idx = np.argsort(smoothed_predicted_scores_instance)
        sort_idx = sort_idx[::-1]  # Sort in descending order
    
        display=calculate_collision_free_suction_candidates(suction_confident_threshold,points_ori, kdtree, normals, smoothed_predicted_scores_instance, sort_idx, voxel_grid, display)


    return display, combined_pcd

if __name__ == "__main__":

    score_model,predictor,dino_model=sim_suction_policy_model(device)

    image_path=demo_path+f"/{demo_number}"+".png"
    pcl_path=demo_path+f"/{demo_number}"+".pcd"
    segmentation_path=demo_path+f"/{demo_number}"+".npz"

    point_cloud, image_pil, image, segmentation = load_data(image_path, pcl_path, segmentation_path)

    with torch.no_grad():
    
        display=[]

        points, points_ori, point_idxs,voxel_grid = preprocess_data(point_cloud, segmentation)

        score_map,camera_res_idx= evaluate_suction_model(points,point_idxs, score_model)

        masks=get_mask_dino_sam(predictor,dino_model,image_pil,image)

        overall_smoothed_score_map = np.zeros_like(score_map)
 
        combined_pcd = o3d.geometry.PointCloud()

        ori_image = np.array(image_pil)

        normals=np.asarray(point_cloud.normals)
        kdtree = o3d.geometry.KDTreeFlann(point_cloud)

        if masks==None:
            print("no target detected")
        else:    
            global_min_score = np.min(score_map)
            global_max_score = np.max(score_map)

            display, combined_pcd = process_masks(suction_confident_threshold,masks, normals, camera_res_idx, points_ori, kdtree, voxel_grid, image_pil, global_min_score, global_max_score)
            
            display.append(combined_pcd)
            
            base_pcd=remove_points_from_pointcloud(point_cloud,combined_pcd)
            
            display.append(base_pcd)
            
            o3d.visualization.draw_geometries_with_custom_animation(display)



