import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml


CLAM_TOOL_DIR = Path(__file__).resolve().parent

# Keep these as strings because downstream code and tests patch them directly.
BASE_DIR = str(CLAM_TOOL_DIR)
INPUT_DIR = str(CLAM_TOOL_DIR / "input")
OUTPUT_DIR = str(CLAM_TOOL_DIR / "output")
TEMP_DIR = str(CLAM_TOOL_DIR / "temp")
MODEL_PATH = str(CLAM_TOOL_DIR / "s_4_checkpoint.pt")


def _run_clam_script(script_name, *args):
    script_path = CLAM_TOOL_DIR / script_name
    cmd = [sys.executable, str(script_path), *args]
    subprocess.run(cmd, check=True, cwd=str(CLAM_TOOL_DIR))


def clean_temp():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    os.makedirs(os.path.join(TEMP_DIR, "patches"), exist_ok=True)
    os.makedirs(os.path.join(TEMP_DIR, "features"), exist_ok=True)


def step1_segmentation(slide_name):
    print(f"\n>>> [Step 1] Segmentation: {slide_name}")
    patch_save_dir = os.path.join(TEMP_DIR, "patches")
    _run_clam_script(
        "create_patches_fp.py",
        "--source", INPUT_DIR,
        "--save_dir", patch_save_dir,
        "--patch_size", "256",
        "--step_size", "256",
        "--preset", "bwh_biopsy.csv",
        "--seg",
        "--patch",
        "--stitch",
    )
    return patch_save_dir


def step2_feature_extraction(patch_dir):
    print(f"\n>>> [Step 2] Feature Extraction")
    feat_save_dir = os.path.join(TEMP_DIR, "features")
    _run_clam_script(
        "extract_features_fp.py",
        "--data_h5_dir", patch_dir,
        "--data_slide_dir", INPUT_DIR,
        "--csv_path", os.path.join(patch_dir, "process_list_autogen.csv"),
        "--feat_dir", feat_save_dir,
        "--batch_size", "512",
        "--slide_ext", ".svs",
    )
    return feat_save_dir


def step3_heatmap_inference(slide_name):
    print(f"\n>>> [Step 3] Inference & Heatmap (Fast Mode)")

    # 1. Prepare CSV
    src_csv = os.path.join(TEMP_DIR, "patches", "process_list_autogen.csv")
    temp_csv_path = os.path.join(TEMP_DIR, "inference_list.csv")
    df = pd.read_csv(src_csv)
    df["process"] = 1
    df.to_csv(temp_csv_path, index=False)

    # 2. Prepare config
    config_dict = {
        "exp_arguments": {
            "n_classes": 2,
            "save_exp_code": "FINAL_DEPLOY",
            "raw_save_dir": os.path.join(TEMP_DIR, "raw_results"),
            "production_save_dir": os.path.join(TEMP_DIR, "prod_results"),
            "batch_size": 2048,
        },
        "data_arguments": {
            "data_dir": INPUT_DIR,
            "data_dir_key": "source",
            "process_list": temp_csv_path,
            "preset": "presets/bwh_biopsy.csv",
            "slide_ext": ".svs",
            "label_dict": {"normal": 0, "tumor": 1},
        },
        "patching_arguments": {
            "patch_size": 256,
            "overlap": 0.5,
            "patch_level": 0,
            "custom_downsample": 1,
        },
        "encoder_arguments": {"model_name": "resnet50_trunc", "target_img_size": 224},
        "model_arguments": {
            "ckpt_path": MODEL_PATH,
            "model_type": "clam_sb",
            "initiate_fn": "initiate_model",
            "model_size": "small",
            "drop_out": 0.25,
            "embed_dim": 1024,
        },
        "heatmap_arguments": {
            "vis_level": 2,
            "alpha": 0.4,
            "blank_canvas": False,
            "save_orig": False,
            "save_ext": "jpg",
            "use_ref_scores": False,
            "blur": False,
            "use_center_shift": True,
            "use_roi": False,
            "calc_heatmap": True,
            "binarize": False,
            "binary_thresh": -1,
            "custom_downsample": 1,
            "cmap": "jet",
        },
        "sample_arguments": {
            "samples": [
                {"name": "topk_high_attention", "sample": True, "seed": 1, "k": 10, "mode": "topk"}
            ]
        },
    }

    yaml_path = os.path.join(TEMP_DIR, "deploy_config.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(config_dict, f)

    _run_clam_script("create_heatmaps.py", "--config", yaml_path)


def organize_output(slide_name):
    print(f"\n>>> [Step 4] Organizing Results & Generating Report")
    slide_id = slide_name.replace(".svs", "")
    os.makedirs(os.path.join(OUTPUT_DIR, "heatmaps"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "report"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "topk_patches"), exist_ok=True)

    results_csv_path = os.path.join(BASE_DIR, "heatmaps", "results", "inference_list.csv")

    diagnosis = "Unknown"
    confidence = "N/A"

    if os.path.exists(results_csv_path):
        try:
            df = pd.read_csv(results_csv_path)

            row = df[df["slide_id"] == slide_name]
            if not row.empty:
                if row.iloc[0]["Pred_0"].lower() == "tumor":
                    diagnosis = "TUMOR (High Risk)"
                    tumor_prob = float(row.iloc[0]["p_0"])
                else:
                    diagnosis = "NORMAL (Low Risk)"
                    if row.iloc[0]["Pred_1"].lower() == "tumor":
                        tumor_prob = float(row.iloc[0]["p_1"])
                    else:
                        tumor_prob = 0.0

                confidence = f"{tumor_prob:.4%}"
                print(f"Diagnosis: {diagnosis} | Confidence: {confidence}")
        except Exception as e:
            print(f"Warning: Failed to parse results CSV: {e}")
    else:
        print(f"Warning: Results CSV not found at {results_csv_path}")

    search_pattern = os.path.join(BASE_DIR, "**", f"*{slide_id}*.jpg")
    found_files = [f for f in glob.glob(search_pattern, recursive=True) if "output" not in f]
    heatmap_dest = "Not Generated"
    if found_files:
        best_file = max(found_files, key=os.path.getsize)
        heatmap_dest = os.path.join(OUTPUT_DIR, "heatmaps", f"{slide_id}_heatmap.jpg")
        shutil.copy(best_file, heatmap_dest)
        print(f"Heatmap Saved: {heatmap_dest}")
    else:
        print("Warning: Heatmap image not found.")

    report_path = os.path.join(OUTPUT_DIR, "report", f"{slide_id}_report.txt")
    with open(report_path, "w") as f:
        f.write("=== AI Pathology Diagnosis Report ===\n")
        f.write(f"Slide ID: {slide_id}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("---------------------------------------\n")
        f.write(f"Prediction: {diagnosis}\n")
        f.write(f"Tumor Confidence: {confidence}\n")
        f.write("---------------------------------------\n")
        f.write("Visual Evidence:\n")
        f.write(f"Heatmap: {heatmap_dest}\n")

    print(f"Report Generated: {report_path}")

    topk_search = os.path.join(TEMP_DIR, "prod_results", "**", "topk_high_attention")
    topk_dirs = glob.glob(topk_search, recursive=True)
    if topk_dirs:
        dest_dir = os.path.join(OUTPUT_DIR, "topk_patches", slide_id)
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        os.makedirs(dest_dir, exist_ok=True)
        for img in glob.glob(os.path.join(topk_dirs[0], "*.png")):
            shutil.copy(img, dest_dir)
        print("Top-K Patches Saved")


if __name__ == "__main__":
    t0 = time.time()
    slides = [f for f in os.listdir(INPUT_DIR) if f.endswith(".svs")]
    if not slides:
        print("Please put a .svs file in 'input' folder!")
        exit()

    target = slides[0]
    try:
        clean_temp()
        step1_segmentation(target)
        # step2_feature_extraction(os.path.join(TEMP_DIR, "patches"))
        step3_heatmap_inference(target)
        organize_output(target)

        print(f"\n>>> [Step 5] Cleaning up workspace")
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            print("Deleted 'temp' folder.")

        heatmaps_trash = os.path.join(BASE_DIR, "heatmaps")
        if os.path.exists(heatmaps_trash):
            shutil.rmtree(heatmaps_trash)
            print("Deleted 'heatmaps' folder.")

        print(f"\n=== Pipeline Done in {time.time() - t0:.1f}s ===")
        print(f"Check clean results in: {OUTPUT_DIR}")

    except Exception as e:
        print(f"Error: {e}")
