import pickle
import glob
from pathlib import Path
import argparse
import sys

base_dir = Path(__file__).parent
parser = argparse.ArgumentParser(description='Get precision for each category')

parser.add_argument('--save_eval_path', type=str, default= (base_dir.parent / "test_similar_eval_result").as_posix(), choices=["../test_similar_inference_result","../test_novel_inference_result"], help='save result path')

args = parser.parse_args()

def compute_average_AP_AVE(folder_path):
    # Use glob to get all pickled result files in the folder
    result_files = glob.glob(f"{folder_path}/*.pkl")
    
    total_AP_values = {
        "top_10%": 0,
        "top_1": 0,
        "top_1%": 0,
        "top_5%": 0
    }

    # Load each file and extract AP values
    for file in result_files:
        with open(file, 'rb') as f:
            candidates = pickle.load(f)
            
            for key in total_AP_values:
                total_AP_values[key] += candidates[key]["AP"]

    # Compute average AP for each category
    num_files = len(result_files)
    average_AP = {key: value / num_files for key, value in total_AP_values.items()}

    return average_AP

def compute_average_AP(folder_path):
    # Use glob to get all pickled result files in the folder
    result_files = glob.glob(f"{folder_path}/*.pkl")
    
    total_values = {
        "top_10%": {"TP": 0, "total_candidates": 0},
        "top_1": {"TP": 0, "total_candidates": 0},
        "top_1%": {"TP": 0, "total_candidates": 0},
        "top_5%": {"TP": 0, "total_candidates": 0}
    }

    # Load each file and extract TP and total candidates values
    for file in result_files:
        with open(file, 'rb') as f:
            candidates = pickle.load(f)
            
            for key in total_values:
                total_values[key]["TP"] += candidates[key]["TP"]
                total_values[key]["total_candidates"] += candidates[key]["total_candidates"]

    # Compute average AP for each category
    average_AP = {}
    for key, values in total_values.items():
        average_AP[key] = values["TP"] / values["total_candidates"] if values["total_candidates"] != 0 else 0

    return average_AP

# Usage
folder_path = args.save_eval_path
average_AP_results = compute_average_AP_AVE(folder_path)
print(average_AP_results)