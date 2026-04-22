import builtins
import importlib.util
from pathlib import Path
from unittest.mock import mock_open

import pandas as pd


def test_organize_output_initializes_missing_output_directories(monkeypatch):
    module_path = Path(__file__).resolve().with_name("pipeline_agent.py")
    spec = importlib.util.spec_from_file_location("clam_pipeline_agent_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    base_dir = "D:/fake-workspace"
    slide_name = "sample_slide.svs"
    slide_id = "sample_slide"
    results_csv_path = f"{base_dir}/heatmaps/results/inference_list.csv"
    heatmap_src = f"{base_dir}/temp/{slide_id}_heatmap.jpg"
    heatmap_dest = f"{base_dir}/output/heatmaps/{slide_id}_heatmap.jpg"
    report_path = f"{base_dir}/output/report/{slide_id}_report.txt"
    topk_dir = f"{base_dir}/temp/prod_results/run_1/topk_high_attention"
    topk_dest_dir = f"{base_dir}/output/topk_patches/{slide_id}"
    normalize = lambda value: value.replace("\\", "/")

    monkeypatch.setattr(module, "BASE_DIR", base_dir)
    monkeypatch.setattr(module, "INPUT_DIR", f"{base_dir}/input")
    monkeypatch.setattr(module, "OUTPUT_DIR", f"{base_dir}/output")
    monkeypatch.setattr(module, "TEMP_DIR", f"{base_dir}/temp")
    monkeypatch.setattr(module, "MODEL_PATH", f"{base_dir}/s_4_checkpoint.pt")
    monkeypatch.setattr(
        module.pd,
        "read_csv",
        lambda path: pd.DataFrame(
            [
                {
                    "slide_id": slide_name,
                    "Pred_0": "tumor",
                    "Pred_1": "normal",
                    "p_0": 0.95,
                    "p_1": 0.05,
                }
            ]
        ),
    )

    def fake_exists(path):
        return normalize(path) == normalize(results_csv_path)

    monkeypatch.setattr(module.os.path, "exists", fake_exists)
    monkeypatch.setattr(
        module.glob,
        "glob",
        lambda pattern, recursive=True: (
            [heatmap_src]
            if pattern.endswith("*.jpg")
            else [f"{topk_dir}/patch_1.png"]
            if pattern.endswith("*.png")
            else [topk_dir]
        ),
    )
    monkeypatch.setattr(module.os.path, "getsize", lambda path: 1 if path == heatmap_src else 2)

    created_dirs = []
    monkeypatch.setattr(
        module.os,
        "makedirs",
        lambda path, exist_ok=False: created_dirs.append((path, exist_ok)),
    )

    copied_paths = []
    monkeypatch.setattr(module.shutil, "copy", lambda src, dst: copied_paths.append((src, dst)))
    monkeypatch.setattr(builtins, "open", mock_open())

    module.organize_output(slide_name)

    normalized_dirs = {(normalize(path), exist_ok) for path, exist_ok in created_dirs}
    normalized_copies = {(normalize(src), normalize(dst)) for src, dst in copied_paths}

    assert (normalize(f"{base_dir}/output/heatmaps"), True) in normalized_dirs
    assert (normalize(f"{base_dir}/output/report"), True) in normalized_dirs
    assert (normalize(f"{base_dir}/output/topk_patches"), True) in normalized_dirs
    assert (normalize(heatmap_src), normalize(heatmap_dest)) in normalized_copies
    assert (normalize(f"{topk_dir}/patch_1.png"), normalize(topk_dest_dir)) in normalized_copies
