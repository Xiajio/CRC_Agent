"""
病理切片分类工具模块 (Pathology CLAM Tools)

基于 CLAM (Clustering-constrained Attention Multiple Instance Learning) 的
全切片图像 (WSI) 分析工具，支持病理切片的肿瘤/正常分类、注意力热力图生成和高关注区域提取。

功能：
1. 病理切片分类 - 判断切片是否包含肿瘤
2. 注意力热力图生成 - 可视化模型关注的区域
3. 高关注区域提取 - 提取最可能存在病变的 Top-K 区域

支持格式：.svs, .tif, .tiff, .ndpi (全切片图像格式)

作者: AI Assistant
版本: v1.0
日期: 2026-01-12
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field


# ==============================================================================
# 路径配置
# ==============================================================================

# CLAM 工具的基础目录
CLAM_TOOL_DIR = os.path.join(
    os.path.dirname(__file__),
    "tool",
    "Pathological_Slide_Classification",
    "CLAM_Tool"
)

# 默认模型路径
DEFAULT_MODEL_PATH = os.path.join(CLAM_TOOL_DIR, "s_4_checkpoint.pt")


# ==============================================================================
# 延迟导入（避免启动时强制依赖）
# ==============================================================================

def _import_torch():
    """延迟导入 PyTorch"""
    try:
        import torch
        return torch
    except ImportError:
        return None


def _import_openslide():
    """延迟导入 OpenSlide"""
    try:
        import openslide
        return openslide
    except ImportError:
        return None


def _import_cv2():
    """延迟导入 OpenCV"""
    try:
        import cv2
        return cv2
    except ImportError:
        return None


def _import_h5py():
    """延迟导入 h5py"""
    try:
        import h5py
        return h5py
    except ImportError:
        return None


def _import_timm():
    """延迟导入 timm"""
    try:
        import timm
        return timm
    except ImportError:
        return None


def _import_yaml():
    """延迟导入 yaml"""
    try:
        import yaml
        return yaml
    except ImportError:
        return None


def _import_pandas():
    """延迟导入 pandas"""
    try:
        import pandas as pd
        return pd
    except ImportError:
        return None


# ==============================================================================
# 模型管理（单例模式）
# ==============================================================================

_clam_model = None
_clam_model_path = None
_encoder_model = None
_device = None


def _get_device():
    """获取计算设备"""
    global _device
    if _device is None:
        torch = _import_torch()
        if torch is not None and torch.cuda.is_available():
            _device = torch.device("cuda")
            print(f"[PathologyCLAM] 使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            _device = torch.device("cpu") if torch else None
            print("[PathologyCLAM] 使用 CPU 模式")
    return _device


def _get_clam_model(model_path: str = None):
    """
    获取或初始化 CLAM 模型（单例模式，懒加载）
    
    Args:
        model_path: 模型文件路径，如果为 None 则使用默认路径
        
    Returns:
        CLAM 模型实例，如果导入失败则返回 None
    """
    global _clam_model, _clam_model_path
    
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    
    # 如果模型已加载且路径相同，直接返回
    if _clam_model is not None and _clam_model_path == model_path:
        return _clam_model
    
    # 检查依赖
    torch = _import_torch()
    if torch is None:
        print("[PathologyCLAM] 警告: torch 模块未安装")
        print("[PathologyCLAM] 请运行: pip install torch")
        return None
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"[PathologyCLAM] 警告: 模型文件不存在: {model_path}")
        return None
    
    try:
        # 添加 CLAM 工具目录到 Python 路径
        if CLAM_TOOL_DIR not in sys.path:
            sys.path.insert(0, CLAM_TOOL_DIR)
        
        # 导入 CLAM 模型
        import importlib
        clam_module = importlib.import_module("models.model_clam")
        CLAM_SB = getattr(clam_module, "CLAM_SB")
        CLAM_MB = getattr(clam_module, "CLAM_MB")
        
        device = _get_device()
        
        print(f"[PathologyCLAM] 正在加载 CLAM 模型: {model_path}")
        
        # 加载模型配置和权重
        checkpoint = torch.load(model_path, map_location=device)
        
        # 根据配置创建模型
        model_args = {
            'gate': True,
            'size_arg': 'small',
            'dropout': 0.25,
            'k_sample': 8,
            'n_classes': 2,
            'embed_dim': 1024
        }
        
        _clam_model = CLAM_SB(**model_args)
        _clam_model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        _clam_model = _clam_model.to(device)
        _clam_model.eval()
        
        _clam_model_path = model_path
        print(f"[PathologyCLAM] CLAM 模型加载成功")
        
        return _clam_model
        
    except Exception as e:
        print(f"[PathologyCLAM] 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def reset_clam_model():
    """重置 CLAM 模型（用于测试或重新初始化）"""
    global _clam_model, _clam_model_path, _encoder_model, _device
    _clam_model = None
    _clam_model_path = None
    _encoder_model = None
    _device = None


def _check_dependencies() -> Dict[str, bool]:
    """检查所有依赖是否已安装"""
    return {
        "torch": _import_torch() is not None,
        "openslide": _import_openslide() is not None,
        "cv2": _import_cv2() is not None,
        "h5py": _import_h5py() is not None,
        "timm": _import_timm() is not None,
        "yaml": _import_yaml() is not None,
        "pandas": _import_pandas() is not None
    }


def _get_missing_dependencies() -> List[str]:
    """获取缺失的依赖列表"""
    deps = _check_dependencies()
    missing = []
    for name, installed in deps.items():
        if not installed:
            if name == "cv2":
                missing.append("opencv-python")
            elif name == "yaml":
                missing.append("pyyaml")
            elif name == "openslide":
                missing.append("openslide-python")
            else:
                missing.append(name)
    return missing


# ==============================================================================
# 辅助函数
# ==============================================================================

def _validate_slide_path(slide_path: str) -> Tuple[bool, str]:
    """验证切片路径"""
    if not slide_path or not slide_path.strip():
        return False, "必须提供切片文件路径"
    
    slide_path = slide_path.strip()
    
    if not os.path.exists(slide_path):
        return False, f"切片文件不存在: {slide_path}"
    
    # 检查文件格式
    supported_formats = ['.svs', '.tif', '.tiff', '.ndpi', '.mrxs', '.vms', '.vmu']
    ext = os.path.splitext(slide_path)[1].lower()
    if ext not in supported_formats:
        return False, f"不支持的文件格式: {ext}，支持的格式: {supported_formats}"
    
    return True, slide_path


def _run_clam_pipeline(
    slide_path: str,
    output_dir: str,
    model_path: str = None,
    generate_heatmap: bool = True,
    extract_topk: bool = True,
    topk: int = 10,
    patch_size: int = 256,
    vis_level: int = 2
) -> Dict:
    """
    运行完整的 CLAM 分析流程
    
    Args:
        slide_path: 切片文件路径
        output_dir: 输出目录
        model_path: 模型路径
        generate_heatmap: 是否生成热力图
        extract_topk: 是否提取 Top-K 区域
        topk: Top-K 数量
        patch_size: 切片大小
        vis_level: 可视化级别
        
    Returns:
        分析结果字典
    """
    yaml = _import_yaml()
    pd = _import_pandas()
    
    if yaml is None or pd is None:
        return {
            "success": False,
            "error_message": "缺少依赖: pyyaml 或 pandas"
        }
    
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="clam_")
    
    try:
        slide_name = os.path.basename(slide_path)
        slide_id = os.path.splitext(slide_name)[0]
        
        # 创建输入目录并复制/链接切片
        input_dir = os.path.join(temp_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        
        # 创建软链接或复制文件
        target_path = os.path.join(input_dir, slide_name)
        try:
            os.symlink(slide_path, target_path)
        except (OSError, NotImplementedError):
            # Windows 可能不支持软链接
            shutil.copy2(slide_path, target_path)
        
        # 创建临时工作目录
        patches_dir = os.path.join(temp_dir, "patches")
        features_dir = os.path.join(temp_dir, "features")
        os.makedirs(patches_dir, exist_ok=True)
        os.makedirs(features_dir, exist_ok=True)
        
        # 保存当前工作目录
        original_cwd = os.getcwd()
        
        try:
            # 切换到 CLAM 工具目录
            os.chdir(CLAM_TOOL_DIR)
            
            # Step 1: 分割和切片
            print(f"[PathologyCLAM] Step 1/3: 组织分割和切片提取...")
            env = os.environ.copy()
            env["PYTHONPATH"] = CLAM_TOOL_DIR + os.pathsep + env.get("PYTHONPATH", "")

            cmd_patch = [
                sys.executable, "create_patches_fp.py",
                "--source", input_dir,
                "--save_dir", patches_dir,
                "--patch_size", str(patch_size),
                "--step_size", str(patch_size),
                "--preset", "bwh_biopsy.csv",
                "--seg", "--patch", "--stitch"
            ]
            result = subprocess.run(cmd_patch, capture_output=True, text=True, timeout=600, env=env)
            if result.returncode != 0:
                print(f"[PathologyCLAM] 切片提取警告: {result.stderr}")
            
            # Step 2: 特征提取
            print(f"[PathologyCLAM] Step 2/3: 深度特征提取...")
            process_csv = os.path.join(patches_dir, "process_list_autogen.csv")
            
            # 获取切片扩展名
            slide_ext = os.path.splitext(slide_name)[1]
            
            cmd_feat = [
                sys.executable, "extract_features_fp.py",
                "--data_h5_dir", patches_dir,
                "--data_slide_dir", input_dir,
                "--csv_path", process_csv,
                "--feat_dir", features_dir,
                "--batch_size", "512",
                "--slide_ext", slide_ext
            ]
            result = subprocess.run(cmd_feat, capture_output=True, text=True, timeout=1200, env=env)
            if result.returncode != 0:
                print(f"[PathologyCLAM] 特征提取警告: {result.stderr}")
            
            # Step 3: 推理和热力图生成
            print(f"[PathologyCLAM] Step 3/3: 模型推理...")
            
            # 准备推理配置
            inference_csv = os.path.join(temp_dir, "inference_list.csv")
            if os.path.exists(process_csv):
                df = pd.read_csv(process_csv)
                df['process'] = 1
                df.to_csv(inference_csv, index=False)
            
            raw_results_dir = os.path.join(temp_dir, "raw_results")
            prod_results_dir = os.path.join(temp_dir, "prod_results")
            
            config_dict = {
                "exp_arguments": {
                    "n_classes": 2,
                    "save_exp_code": "INFERENCE",
                    "raw_save_dir": raw_results_dir,
                    "production_save_dir": prod_results_dir,
                    "batch_size": 2048
                },
                "data_arguments": {
                    "data_dir": input_dir,
                    "data_dir_key": "source",
                    "process_list": inference_csv,
                    "preset": "presets/bwh_biopsy.csv",
                    "slide_ext": slide_ext,
                    "label_dict": {"normal": 0, "tumor": 1}
                },
                "patching_arguments": {
                    "patch_size": patch_size,
                    "overlap": 0.5,
                    "patch_level": 0,
                    "custom_downsample": 1
                },
                "encoder_arguments": {
                    "model_name": "resnet50_trunc",
                    "target_img_size": 224
                },
                "model_arguments": {
                    "ckpt_path": model_path,
                    "model_type": "clam_sb",
                    "initiate_fn": "initiate_model",
                    "model_size": "small",
                    "drop_out": 0.25,
                    "embed_dim": 1024
                },
                "heatmap_arguments": {
                    "vis_level": vis_level,
                    "alpha": 0.4,
                    "blank_canvas": False,
                    "save_orig": False,
                    "save_ext": "jpg",
                    "use_ref_scores": False,
                    "blur": False,
                    "use_center_shift": True,
                    "use_roi": False,
                    "calc_heatmap": generate_heatmap,
                    "binarize": False,
                    "binary_thresh": -1,
                    "custom_downsample": 1,
                    "cmap": "jet"
                },
                "sample_arguments": {
                    "samples": [
                        {
                            "name": "topk_high_attention",
                            "sample": extract_topk,
                            "seed": 1,
                            "k": topk,
                            "mode": "topk"
                        }
                    ]
                }
            }
            
            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f)
            
            cmd_heatmap = [
                sys.executable, "create_heatmaps.py",
                "--config", config_path
            ]
            result = subprocess.run(cmd_heatmap, capture_output=True, text=True, timeout=1800, env=env)
            if result.returncode != 0:
                print(f"[PathologyCLAM] 热力图生成警告: {result.stderr}")
            
        finally:
            # 恢复工作目录
            os.chdir(original_cwd)
        
        # 收集结果
        results = {
            "success": True,
            "slide_id": slide_id,
            "slide_path": slide_path,
            "prediction": "unknown",
            "tumor_probability": None,
            "normal_probability": None,
            "confidence": None,
            "heatmap_path": None,
            "topk_patches_dir": None,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        # 解析推理结果
        results_csv = os.path.join(CLAM_TOOL_DIR, "heatmaps", "results", "inference_list.csv")
        if os.path.exists(results_csv):
            try:
                df = pd.read_csv(results_csv)
                row = df[df['slide_id'] == slide_name]
                if not row.empty:
                    pred_0 = str(row.iloc[0].get('Pred_0', '')).lower()
                    p_0 = float(row.iloc[0].get('p_0', 0))
                    p_1 = float(row.iloc[0].get('p_1', 0))
                    
                    if pred_0 == 'tumor':
                        results["prediction"] = "TUMOR"
                        results["tumor_probability"] = round(p_0, 4)
                        results["normal_probability"] = round(p_1, 4)
                        results["confidence"] = round(p_0, 4)
                    else:
                        results["prediction"] = "NORMAL"
                        results["tumor_probability"] = round(p_1, 4)
                        results["normal_probability"] = round(p_0, 4)
                        results["confidence"] = round(p_0, 4)
            except Exception as e:
                print(f"[PathologyCLAM] 解析结果警告: {e}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        heatmap_output_dir = os.path.join(output_dir, "heatmaps")
        topk_output_dir = os.path.join(output_dir, "topk_patches")
        report_output_dir = os.path.join(output_dir, "reports")
        os.makedirs(heatmap_output_dir, exist_ok=True)
        os.makedirs(topk_output_dir, exist_ok=True)
        os.makedirs(report_output_dir, exist_ok=True)
        
        # 复制热力图
        import glob
        heatmap_search = os.path.join(CLAM_TOOL_DIR, "**", f"*{slide_id}*.jpg")
        found_heatmaps = [f for f in glob.glob(heatmap_search, recursive=True) if output_dir not in f]
        if found_heatmaps:
            best_heatmap = max(found_heatmaps, key=os.path.getsize)
            heatmap_dest = os.path.join(heatmap_output_dir, f"{slide_id}_heatmap.jpg")
            shutil.copy2(best_heatmap, heatmap_dest)
            results["heatmap_path"] = heatmap_dest
        
        # 复制 Top-K 切片
        topk_search = os.path.join(prod_results_dir, "**", "topk_high_attention")
        topk_dirs = glob.glob(topk_search, recursive=True)
        if topk_dirs:
            topk_dest = os.path.join(topk_output_dir, slide_id)
            if os.path.exists(topk_dest):
                shutil.rmtree(topk_dest)
            os.makedirs(topk_dest, exist_ok=True)
            
            patches_copied = 0
            for img in glob.glob(os.path.join(topk_dirs[0], "*.png")):
                shutil.copy2(img, topk_dest)
                patches_copied += 1
            
            if patches_copied > 0:
                results["topk_patches_dir"] = topk_dest
                results["topk_patches_count"] = patches_copied
        
        # 生成报告
        report_path = os.path.join(report_output_dir, f"{slide_id}_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write("    AI 病理诊断报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"切片 ID: {slide_id}\n")
            f.write(f"分析时间: {results['processing_timestamp']}\n")
            f.write("-" * 50 + "\n\n")
            f.write(f"预测结果: {results['prediction']}\n")
            f.write(f"肿瘤概率: {results['tumor_probability']}\n")
            f.write(f"正常概率: {results['normal_probability']}\n")
            f.write("-" * 50 + "\n\n")
            f.write("可视化证据:\n")
            f.write(f"  热力图: {results.get('heatmap_path', '未生成')}\n")
            f.write(f"  高关注区域: {results.get('topk_patches_dir', '未生成')}\n")
            f.write("=" * 50 + "\n")
        
        results["report_path"] = report_path
        
        return results
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error_message": "处理超时，请检查切片文件大小或系统资源"
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error_message": f"处理失败: {str(e)}",
            "traceback": traceback.format_exc()
        }
    finally:
        # 清理临时目录
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        
        # 清理 CLAM 工具目录下的临时文件
        try:
            heatmaps_dir = os.path.join(CLAM_TOOL_DIR, "heatmaps")
            if os.path.exists(heatmaps_dir):
                shutil.rmtree(heatmaps_dir)
        except:
            pass


# ==============================================================================
# LangChain Tool 定义
# ==============================================================================

@tool
def pathology_slide_classify(
    slide_path: str,
    output_dir: str = None,
    model_path: str = None,
    generate_heatmap: bool = True,
    extract_topk: bool = True,
    topk: int = 10
) -> Dict:
    """
    病理切片分类工具 - 使用 CLAM 模型对全切片图像进行肿瘤/正常分类
    
    功能说明：
    - 对 .svs 等格式的病理全切片图像进行 AI 分析
    - 自动分割组织区域并提取特征
    - 使用注意力机制定位可疑区域
    - 输出分类结果、热力图和高关注区域切片
    
    输入参数：
    - slide_path: 病理切片文件路径（支持 .svs, .tif, .tiff, .ndpi 格式）
    - output_dir: 输出目录路径（可选，默认在切片目录旁创建 results 目录）
    - model_path: CLAM 模型路径（可选，使用默认模型）
    - generate_heatmap: 是否生成注意力热力图（默认 True）
    - extract_topk: 是否提取高关注区域切片（默认 True）
    - topk: 提取的高关注区域数量（默认 10）
    
    输出说明：
    返回包含以下信息的字典：
    - success: 是否成功完成分析
    - slide_id: 切片标识符
    - prediction: 预测结果（TUMOR/NORMAL）
    - tumor_probability: 肿瘤概率
    - normal_probability: 正常概率
    - confidence: 置信度
    - heatmap_path: 热力图文件路径
    - topk_patches_dir: Top-K 切片目录
    - report_path: 分析报告路径
    - error_message: 如果失败，包含错误信息
    
    使用示例：
    - pathology_slide_classify(slide_path="/data/slides/sample.svs")
    - pathology_slide_classify(slide_path="/data/slides/sample.svs", output_dir="/results", topk=20)
    
    注意事项：
    - 首次运行需要下载 ResNet50 预训练权重
    - 处理大型切片文件可能需要较长时间（几分钟到十几分钟）
    - 需要 openslide-python 库来读取 .svs 文件
    - GPU 可以显著加速处理过程
    """
    
    # 验证切片路径
    valid, result = _validate_slide_path(slide_path)
    if not valid:
        return {
            "success": False,
            "error_message": result
        }
    slide_path = result
    
    # 检查依赖
    missing = _get_missing_dependencies()
    if missing:
        return {
            "success": False,
            "error_message": f"缺少依赖库: {', '.join(missing)}",
            "suggestion": f"请运行: pip install {' '.join(missing)}"
        }
    
    # 设置默认输出目录
    if output_dir is None:
        slide_dir = os.path.dirname(slide_path)
        slide_name = os.path.splitext(os.path.basename(slide_path))[0]
        output_dir = os.path.join(slide_dir, f"{slide_name}_clam_results")
    
    print(f"\n{'='*60}")
    print(f"[病理切片分类工具] 开始处理")
    print(f"切片路径: {slide_path}")
    print(f"输出目录: {output_dir}")
    print(f"生成热力图: {generate_heatmap}")
    print(f"提取 Top-K: {extract_topk} (K={topk})")
    print(f"{'='*60}\n")
    
    # 运行 CLAM 分析流程
    result = _run_clam_pipeline(
        slide_path=slide_path,
        output_dir=output_dir,
        model_path=model_path,
        generate_heatmap=generate_heatmap,
        extract_topk=extract_topk,
        topk=topk
    )
    
    if result.get("success"):
        print(f"\n{'='*60}")
        print(f"[病理切片分类工具] 分析完成")
        print(f"预测结果: {result.get('prediction')}")
        print(f"肿瘤概率: {result.get('tumor_probability')}")
        print(f"输出目录: {output_dir}")
        print(f"{'='*60}\n")
    
    return result


@tool
def quick_pathology_check(slide_path: str) -> Dict:
    """
    快速病理检查工具 - 快速检测切片是否可能包含肿瘤
    
    功能说明：
    - 对病理切片进行快速分析
    - 仅返回分类结果，不生成热力图和切片
    - 适用于大批量筛选场景
    
    输入参数：
    - slide_path: 病理切片文件路径
    
    输出说明：
    返回包含以下信息的字典：
    - success: 是否成功
    - prediction: 预测结果（TUMOR/NORMAL）
    - tumor_probability: 肿瘤概率
    - confidence: 置信度
    
    使用示例：
    - quick_pathology_check(slide_path="/data/slides/sample.svs")
    """
    
    # 验证切片路径
    valid, result = _validate_slide_path(slide_path)
    if not valid:
        return {
            "success": False,
            "error_message": result
        }
    
    # 使用临时目录
    with tempfile.TemporaryDirectory(prefix="clam_quick_") as temp_output:
        result = _run_clam_pipeline(
            slide_path=result,
            output_dir=temp_output,
            generate_heatmap=False,
            extract_topk=False,
            topk=0
        )
        
        # 只返回关键信息
        return {
            "success": result.get("success", False),
            "slide_id": result.get("slide_id"),
            "prediction": result.get("prediction"),
            "tumor_probability": result.get("tumor_probability"),
            "normal_probability": result.get("normal_probability"),
            "confidence": result.get("confidence"),
            "processing_timestamp": result.get("processing_timestamp"),
            "error_message": result.get("error_message")
        }


@tool
def get_pathology_clam_status() -> Dict:
    """
    获取病理 CLAM 工具状态信息
    
    功能说明：
    - 检查模型是否已加载
    - 检查依赖是否已安装
    - 返回工具版本和配置信息
    
    输出说明：
    返回包含以下信息的字典：
    - model_loaded: 模型是否已加载
    - model_path: 当前/默认模型路径
    - model_exists: 模型文件是否存在
    - dependencies: 依赖检查结果
    - missing_dependencies: 缺失的依赖列表
    - tool_version: 工具版本
    - supported_formats: 支持的切片格式
    
    使用示例：
    - get_pathology_clam_status()
    """
    
    global _clam_model, _clam_model_path
    
    deps = _check_dependencies()
    missing = _get_missing_dependencies()
    
    # 检查 GPU
    torch = _import_torch()
    gpu_available = False
    gpu_name = None
    if torch is not None:
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
    
    return {
        "model_loaded": _clam_model is not None,
        "model_path": _clam_model_path or DEFAULT_MODEL_PATH,
        "model_exists": os.path.exists(DEFAULT_MODEL_PATH),
        "clam_tool_dir": CLAM_TOOL_DIR,
        "clam_tool_exists": os.path.exists(CLAM_TOOL_DIR),
        "dependencies": deps,
        "missing_dependencies": missing,
        "all_dependencies_installed": len(missing) == 0,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "tool_version": "1.0.0",
        "supported_formats": [".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".vms", ".vmu"],
        "description": "基于 CLAM 的病理全切片图像分类工具",
        "functions": {
            "pathology_slide_classify": "对病理切片进行肿瘤/正常分类，生成热力图和报告",
            "quick_pathology_check": "快速检测切片是否包含肿瘤",
            "get_pathology_clam_status": "获取工具状态信息"
        }
    }


# ==============================================================================
# 高级聚合工具
# ==============================================================================

def perform_comprehensive_pathology_analysis(patient_id: str) -> Dict:
    """
    高级聚合工具：输入患者 ID，自动执行完整的病理切片分析流程
    
    功能说明：
    - 输入患者 ID（如 "93" 或 "093"）
    - 自动查找对应的病理切片文件
    - 执行 CLAM 分析
    - 返回汇总的诊断报告
    
    输入参数：
    - patient_id: 患者编号
    
    输出说明：
    返回包含以下信息的字典：
    - patient_id: 患者 ID
    - slides_analyzed: 分析的切片数量
    - results: 每个切片的分析结果
    - summary: 汇总诊断
    
    使用示例：
    - perform_comprehensive_pathology_analysis(patient_id="93")
    
    适用场景：
    - 病理诊断请求
    - 组织病变分析
    - 癌症分型判断
    """
    
    if not patient_id:
        return {
            "success": False,
            "error_message": "必须提供患者 ID"
        }
    
    patient_id_str = str(patient_id).strip()
    if patient_id_str.isdigit():
        folder_name = patient_id_str.zfill(3)
    else:
        folder_name = patient_id_str
    
    # 查找病理切片目录（可根据实际情况调整路径）
    base_paths = [
        Path(r"E:/LangG/data/Case Database/Silds"),
        Path(r"E:/LangG/data/Case Database/Pathological Slides"),
        Path(r"E:/LangG/data/pathology"),
    ]
    
    slides = []
    for base_path in base_paths:
        if base_path.exists():
            patient_dir = base_path / folder_name
            if patient_dir.exists():
                for ext in ["*.svs", "*.tif", "*.tiff", "*.ndpi"]:
                    slides.extend(list(patient_dir.glob(ext)))
    
    if not slides:
        return {
            "success": False,
            "error_message": f"未找到患者 {folder_name} 的病理切片文件",
            "patient_id": folder_name,
            "searched_paths": [str(p) for p in base_paths]
        }
    
    print(f"[PathologyCLAM] 开始分析患者 {folder_name} 的 {len(slides)} 个切片...")
    
    results = []
    for slide in slides:
        result = pathology_slide_classify.invoke({
            "slide_path": str(slide),
            "generate_heatmap": True,
            "extract_topk": True
        })
        results.append(result)
    
    # 汇总分析
    tumor_count = sum(1 for r in results if r.get("prediction") == "TUMOR")
    normal_count = sum(1 for r in results if r.get("prediction") == "NORMAL")
    
    summary = {
        "success": True,
        "patient_id": folder_name,
        "slides_analyzed": len(slides),
        "tumor_slides": tumor_count,
        "normal_slides": normal_count,
        "overall_diagnosis": "TUMOR" if tumor_count > 0 else "NORMAL",
        "results": results,
        "processing_timestamp": datetime.now().isoformat()
    }
    
    return summary


# ==============================================================================
# 工具注册
# ==============================================================================

class PerformPathologyAnalysisInput(BaseModel):
    """perform_comprehensive_pathology_analysis 的输入参数"""
    patient_id: str = Field(
        ...,
        description="患者编号"
    )


def _execute_pathology_analysis(patient_id: str) -> Dict:
    """内部执行函数"""
    return perform_comprehensive_pathology_analysis(patient_id)


# 创建 LangChain StructuredTool
perform_comprehensive_pathology_analysis_tool = StructuredTool(
    func=_execute_pathology_analysis,
    name="perform_comprehensive_pathology_analysis",
    description="高级聚合工具：输入患者 ID，自动执行完整的病理切片分析。触发条件：用户需要病理诊断、组织病变分析、癌症分型。触发关键词：病理分析、切片诊断、组织学检查、病理报告",
    args_schema=PerformPathologyAnalysisInput
)


def list_pathology_clam_tools():
    """
    返回病理 CLAM 工具列表
    
    Returns:
        list: 包含所有病理 CLAM 工具的列表
    """
    return [
        pathology_slide_classify,
        quick_pathology_check,
        get_pathology_clam_status,
        perform_comprehensive_pathology_analysis
    ]


def get_pathology_clam_tools():
    """
    获取病理 CLAM 工具实例列表（用于智能体工具注册）
    
    Returns:
        list: LangChain 工具实例列表
    """
    return [
        pathology_slide_classify,
        quick_pathology_check,
        get_pathology_clam_status,
        perform_comprehensive_pathology_analysis_tool
    ]


# ==============================================================================
# 模块测试
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("病理 CLAM 工具模块测试")
    print("=" * 60)
    
    # 检查工具状态
    status = get_pathology_clam_status.invoke({})
    print("\n工具状态:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\n可用工具:")
    for tool in list_pathology_clam_tools():
        if hasattr(tool, 'name'):
            print(f"  - {tool.name}")
        else:
            print(f"  - {tool.__name__}")
