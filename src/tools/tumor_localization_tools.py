"""
肿瘤定位工具模块

提供基于U-Net模型的CT图像肿瘤分割和定位功能。
用于精确定位肿瘤在CT图像中的位置，并生成分割掩码。
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

# 注意：torch 和相关库使用延迟导入
# 这样即使这些包没安装，智能体也能启动，只是不能使用肿瘤定位功能

# 尝试加载模型（延迟初始化）
_localization_model = None
_model_path = None
_device = None


def _import_torch():
    """延迟导入 torch 模块"""
    try:
        import torch
        return torch
    except ImportError:
        return None


def _import_cv2():
    """延迟导入 cv2 模块"""
    try:
        import cv2
        return cv2
    except ImportError:
        return None


def _import_pil():
    """延迟导入 PIL 模块"""
    try:
        from PIL import Image
        return Image
    except ImportError:
        return None


def get_localization_model(model_path: str = None):
    """
    获取或初始化U-Net肿瘤定位模型（单例模式，懒加载）
    
    Args:
        model_path: 模型权重文件路径，如果为None则使用默认路径
        
    Returns:
        模型实例，如果导入失败则返回None
    """
    global _localization_model, _model_path, _device
    
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(__file__),
            "tool",
            "Tumor_Localization",
            "checkpoint_epoch_last.pth"
        )
    
    # 如果模型已加载且路径相同，直接返回
    if _localization_model is not None and _model_path == model_path:
        return _localization_model
    
    # 延迟导入 torch
    torch = _import_torch()
    if torch is None:
        print("[TumorLocalization] 警告: torch 模块未安装，无法加载模型")
        print("[TumorLocalization] 请运行: pip install torch torchvision")
        return None
    
    try:
        # 检查CUDA可用性并设置设备
        if torch.cuda.is_available():
            _device = torch.device('cuda')
            print(f"[TumorLocalization] 检测到CUDA GPU: {torch.cuda.get_device_name(0)}")
            print(f"[TumorLocalization] GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            _device = torch.device('cpu')
            print("[TumorLocalization] 未检测到CUDA，使用CPU模式")
        
        # 导入模型架构
        model_dir = os.path.join(
            os.path.dirname(__file__),
            "tool",
            "Tumor_Localization"
        )
        
        # 动态导入模型模块
        import sys
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        
        from Tumor_Localization import create_tumor_localization_model
        
        print(f"[TumorLocalization] 正在加载U-Net模型: {model_path}")
        
        # 创建模型实例
        _localization_model = create_tumor_localization_model(
            n_channels=3,  # RGB图像
            n_classes=2,   # 二分类：背景和肿瘤
            bilinear=False
        )
        
        # 加载训练好的权重
        checkpoint = torch.load(model_path, map_location=_device)
        
        # 处理不同的checkpoint格式
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"[TumorLocalization] 加载checkpoint，训练epoch: {checkpoint.get('epoch', 'unknown')}")
        else:
            state_dict = checkpoint
        
        # 过滤掉非模型权重的额外键（如 mask_values）
        model_keys = set(_localization_model.state_dict().keys())
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
        
        # 检查是否有被过滤的键
        extra_keys = set(state_dict.keys()) - model_keys
        if extra_keys:
            print(f"[TumorLocalization] 忽略checkpoint中的额外键: {extra_keys}")
        
        _localization_model.load_state_dict(filtered_state_dict)
        
        # 将模型移动到目标设备并设置为评估模式
        _localization_model = _localization_model.to(_device)
        _localization_model.eval()
        
        _model_path = model_path
        print(f"[TumorLocalization] 模型加载成功 (设备: {_device})")
        
        return _localization_model
        
    except Exception as e:
        print(f"[TumorLocalization] 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def reset_localization_model():
    """重置模型（用于测试或重新初始化）"""
    global _localization_model, _model_path, _device
    _localization_model = None
    _model_path = None
    _device = None


def preprocess_image(image_path: str, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    预处理输入图像
    
    Args:
        image_path: 图像文件路径
        target_size: 目标尺寸 (height, width)
        
    Returns:
        预处理后的图像数组 (C, H, W)，归一化到[0, 1]
    """
    cv2 = _import_cv2()
    if cv2 is None:
        raise ImportError("cv2模块未安装")
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 转换为RGB（OpenCV默认是BGR）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 调整大小
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # 归一化到[0, 1]
    image = image.astype(np.float32) / 255.0
    
    # 转换为CHW格式
    image = np.transpose(image, (2, 0, 1))
    
    return image


def postprocess_mask(mask: np.ndarray, original_size: Tuple[int, int], 
                     threshold: float = 0.5) -> np.ndarray:
    """
    后处理分割掩码
    
    Args:
        mask: 模型输出的掩码 (H, W)
        original_size: 原始图像尺寸 (height, width)
        threshold: 二值化阈值
        
    Returns:
        处理后的掩码 (H, W)，值为0或255
    """
    cv2 = _import_cv2()
    if cv2 is None:
        raise ImportError("cv2模块未安装")
    
    # 应用阈值
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # 调整到原始尺寸
    binary_mask = cv2.resize(binary_mask, (original_size[1], original_size[0]), 
                            interpolation=cv2.INTER_NEAREST)
    
    # 转换为0-255范围
    binary_mask = binary_mask * 255
    
    return binary_mask


def extract_tumor_info(mask: np.ndarray) -> Dict:
    """
    从分割掩码中提取肿瘤信息
    
    Args:
        mask: 二值化掩码 (H, W)
        
    Returns:
        包含肿瘤位置、大小等信息的字典
    """
    cv2 = _import_cv2()
    if cv2 is None:
        raise ImportError("cv2模块未安装")
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {
            "has_tumor": False,
            "tumor_count": 0,
            "total_area": 0,
            "regions": []
        }
    
    # 计算每个肿瘤区域的信息
    regions = []
    total_area = 0
    
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 10:  # 过滤太小的区域
            continue
            
        total_area += area
        
        # 计算边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 计算中心点
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2
        
        regions.append({
            "region_id": idx + 1,
            "area": int(area),
            "bounding_box": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            },
            "center": {
                "x": int(cx),
                "y": int(cy)
            },
            "perimeter": int(cv2.arcLength(contour, True))
        })
    
    # 按面积排序
    regions.sort(key=lambda r: r["area"], reverse=True)
    
    return {
        "has_tumor": len(regions) > 0,
        "tumor_count": len(regions),
        "total_area": int(total_area),
        "regions": regions
    }


# ==============================================================================
# LangChain Tool 定义
# ==============================================================================

@tool
def tumor_localization_tool(
    image_path: str,
    output_path: str = None,
    model_path: str = None,
    threshold: float = 0.5,
    save_visualization: bool = True
) -> Dict:
    """
    肿瘤定位工具 - 使用U-Net模型对CT图像进行肿瘤分割和定位
    
    功能说明：
    - 对单张CT图像进行精确的肿瘤分割
    - 生成肿瘤区域的分割掩码
    - 提取肿瘤位置、大小、边界框等信息
    - 可选生成可视化结果图像
    
    输入参数：
    - image_path: 输入CT图像路径
    - output_path: 输出掩码图像路径（可选，默认在原图旁保存）
    - model_path: U-Net模型权重文件路径（可选，使用默认模型）
    - threshold: 分割阈值，0-1之间（默认0.5）
    - save_visualization: 是否保存可视化结果（默认True）
    
    输出说明：
    返回包含以下信息的字典：
    - success: 是否成功完成分割
    - has_tumor: 是否检测到肿瘤
    - tumor_count: 检测到的肿瘤区域数量
    - total_area: 肿瘤总面积（像素）
    - regions: 每个肿瘤区域的详细信息列表
    - mask_path: 保存的掩码图像路径
    - visualization_path: 可视化结果图像路径
    - processing_timestamp: 处理时间戳
    - error_message: 如果失败，包含错误信息
    
    使用示例：
    - tumor_localization_tool(image_path="/path/to/ct_image.png")
    - tumor_localization_tool(image_path="/data/ct.jpg", threshold=0.6, save_visualization=True)
    
    注意事项：
    - 模型权重文件需要预先训练并放置在正确位置
    - 输入图像会被自动调整为256x256进行推理
    - 输出掩码会调整回原始图像尺寸
    """
    
    # 参数验证
    if not image_path or not image_path.strip():
        return {
            "success": False,
            "error_message": "错误：必须提供图像路径 (image_path)",
            "example_usage": "tumor_localization_tool(image_path='/path/to/ct_image.png')"
        }
    
    image_path = image_path.strip()
    
    # 检查图像文件是否存在
    if not os.path.exists(image_path):
        return {
            "success": False,
            "error_message": f"错误：图像文件不存在: {image_path}",
            "suggestion": "请检查路径是否正确"
        }
    
    # 获取模型
    model = get_localization_model(model_path)
    if model is None:
        return {
            "success": False,
            "error_message": "错误：无法加载肿瘤定位模型",
            "model_path": model_path or os.path.join(os.path.dirname(__file__), 
                                                      "tool", "Tumor_Localization", 
                                                      "checkpoint_epoch_last.pth"),
            "suggestion": "请确保模型权重文件存在于正确的目录中"
        }
    
    try:
        torch = _import_torch()
        cv2 = _import_cv2()
        
        if torch is None or cv2 is None:
            return {
                "success": False,
                "error_message": "依赖模块未安装",
                "suggestion": "请运行: pip install torch opencv-python"
            }
        
        print(f"\n{'='*60}")
        print(f"[肿瘤定位工具] 开始处理")
        print(f"输入图像: {image_path}")
        print(f"分割阈值: {threshold}")
        print(f"{'='*60}\n")
        
        # 读取原始图像尺寸
        original_image = cv2.imread(image_path)
        if original_image is None:
            return {
                "success": False,
                "error_message": f"无法读取图像: {image_path}"
            }
        
        original_size = original_image.shape[:2]  # (height, width)
        print(f"原始图像尺寸: {original_size[1]} x {original_size[0]}")
        
        # 预处理图像
        print("预处理图像...")
        input_tensor = preprocess_image(image_path, target_size=(256, 256))
        
        # 转换为tensor并添加batch维度
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(_device)
        
        # 模型推理
        print("执行模型推理...")
        with torch.no_grad():
            output = model(input_tensor)
            
            # 应用softmax获取概率
            probs = torch.softmax(output, dim=1)
            
            # 获取肿瘤类别的概率（通道1）
            tumor_prob = probs[0, 1, :, :].cpu().numpy()
        
        print(f"推理完成，最大概率: {tumor_prob.max():.4f}")
        
        # 后处理掩码
        print("后处理分割掩码...")
        binary_mask = postprocess_mask(tumor_prob, original_size, threshold)
        
        # 提取肿瘤信息
        print("提取肿瘤信息...")
        tumor_info = extract_tumor_info(binary_mask)
        
        # 设置输出路径
        if output_path is None:
            base_name = os.path.splitext(image_path)[0]
            output_path = f"{base_name}_mask.png"
        
        # 保存掩码
        cv2.imwrite(output_path, binary_mask)
        print(f"掩码已保存: {output_path}")
        
        result = {
            "success": True,
            "has_tumor": tumor_info["has_tumor"],
            "tumor_count": tumor_info["tumor_count"],
            "total_area": tumor_info["total_area"],
            "regions": tumor_info["regions"],
            "mask_path": output_path,
            "input_image": image_path,
            "image_size": {
                "width": original_size[1],
                "height": original_size[0]
            },
            "threshold_used": threshold,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        # 生成可视化结果
        if save_visualization and tumor_info["has_tumor"]:
            visualization_path = f"{os.path.splitext(image_path)[0]}_visualization.png"
            
            # 创建叠加图像
            overlay = original_image.copy()
            
            # 在原图上绘制肿瘤轮廓
            mask_3channel = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 绘制轮廓和边界框
            for region in tumor_info["regions"]:
                bbox = region["bounding_box"]
                center = region["center"]
                
                # 绘制边界框（绿色）
                cv2.rectangle(overlay, 
                            (bbox["x"], bbox["y"]), 
                            (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]),
                            (0, 255, 0), 2)
                
                # 绘制中心点（红色）
                cv2.circle(overlay, (center["x"], center["y"]), 5, (0, 0, 255), -1)
                
                # 添加标签
                label = f"Region {region['region_id']}: {region['area']}px"
                cv2.putText(overlay, label, 
                           (bbox["x"], bbox["y"] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 绘制半透明掩码
            colored_mask = np.zeros_like(original_image)
            colored_mask[binary_mask > 0] = [255, 0, 0]  # 红色掩码
            overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
            
            cv2.imwrite(visualization_path, overlay)
            result["visualization_path"] = visualization_path
            print(f"可视化结果已保存: {visualization_path}")
        
        # 打印总结
        print(f"\n{'='*60}")
        print(f"[肿瘤定位工具] 处理完成")
        if tumor_info["has_tumor"]:
            print(f"检测到 {tumor_info['tumor_count']} 个肿瘤区域")
            print(f"总面积: {tumor_info['total_area']} 像素")
        else:
            print("未检测到肿瘤")
        print(f"{'='*60}\n")
        
        return result
        
    except Exception as e:
        error_msg = f"定位过程发生错误: {str(e)}"
        print(f"[肿瘤定位工具] 错误: {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error_message": error_msg,
            "image_path": image_path
        }


@tool
def batch_tumor_localization(
    input_dir: str,
    output_dir: str = None,
    model_path: str = None,
    threshold: float = 0.5,
    image_extensions: List[str] = None
) -> Dict:
    """
    批量肿瘤定位工具 - 对目录中的所有CT图像进行批量分割和定位
    
    功能说明：
    - 遍历指定目录下的所有图像
    - 对每张图像进行肿瘤分割
    - 保持原始文件夹结构
    - 生成批量处理报告
    
    输入参数：
    - input_dir: 输入图像目录路径
    - output_dir: 输出目录路径（可选）
    - model_path: U-Net模型权重文件路径（可选）
    - threshold: 分割阈值（默认0.5）
    - image_extensions: 图像文件扩展名列表（可选）
    
    输出说明：
    返回包含以下信息的字典：
    - success: 是否成功完成批量处理
    - total_images: 处理的总图像数量
    - images_with_tumor: 检测到肿瘤的图像数量
    - images_without_tumor: 未检测到肿瘤的图像数量
    - tumor_detection_rate: 肿瘤检出率
    - output_directory: 输出目录路径
    - processing_timestamp: 处理时间戳
    - sample_results: 样本结果列表（最多10个）
    
    使用示例：
    - batch_tumor_localization(input_dir="/path/to/ct_images")
    - batch_tumor_localization(input_dir="/data/CT", output_dir="/results/masks", threshold=0.6)
    """
    
    # 参数验证
    if not input_dir or not input_dir.strip():
        return {
            "success": False,
            "error_message": "错误：必须提供输入目录路径 (input_dir)"
        }
    
    input_dir = input_dir.strip()
    
    # 设置默认图像扩展名
    if image_extensions is None:
        image_extensions = ['.png', '.jpg', '.jpeg']
    
    # 设置默认输出目录
    if output_dir is None:
        base_name = os.path.basename(input_dir.rstrip(os.sep))
        output_dir = os.path.join(
            os.path.dirname(input_dir),
            f"{base_name}_localized"
        )
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        return {
            "success": False,
            "error_message": f"错误：输入目录不存在: {input_dir}"
        }
    
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"[批量肿瘤定位] 开始处理")
        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")
        print(f"{'='*60}\n")
        
        # 收集所有图像文件
        all_image_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                file_lower = file.lower()
                if any(file_lower.endswith(ext) for ext in image_extensions):
                    all_image_files.append(os.path.join(root, file))
        
        total_images = len(all_image_files)
        print(f"找到 {total_images} 张图像")
        
        if total_images == 0:
            return {
                "success": False,
                "error_message": f"在目录 {input_dir} 中未找到支持的图像文件",
                "supported_formats": image_extensions
            }
        
        # 批量处理
        images_with_tumor = 0
        images_without_tumor = 0
        sample_results = []
        
        for idx, image_path in enumerate(all_image_files):
            print(f"\n处理 [{idx+1}/{total_images}]: {os.path.basename(image_path)}")
            
            # 计算相对路径
            relative_path = os.path.relpath(os.path.dirname(image_path), input_dir)
            output_subdir = os.path.join(output_dir, relative_path)
            os.makedirs(output_subdir, exist_ok=True)
            
            # 设置输出路径
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_mask_path = os.path.join(output_subdir, f"{base_name}_mask.png")
            
            # 执行定位
            result = tumor_localization_tool.invoke({
                "image_path": image_path,
                "output_path": output_mask_path,
                "model_path": model_path,
                "threshold": threshold,
                "save_visualization": False  # 批量处理时不保存可视化
            })
            
            if isinstance(result, dict) and result.get("success"):
                if result.get("has_tumor"):
                    images_with_tumor += 1
                    if len(sample_results) < 10:
                        sample_results.append({
                            "image_path": image_path,
                            "tumor_count": result.get("tumor_count"),
                            "total_area": result.get("total_area"),
                            "mask_path": output_mask_path
                        })
                else:
                    images_without_tumor += 1
        
        # 计算检出率
        detection_rate = (images_with_tumor / total_images * 100) if total_images > 0 else 0
        
        # 生成报告
        report = {
            "success": True,
            "summary": {
                "total_images_processed": total_images,
                "images_with_tumor": images_with_tumor,
                "images_without_tumor": images_without_tumor,
                "tumor_detection_rate": f"{detection_rate:.1f}%",
                "threshold_used": threshold
            },
            "output_directory": output_dir,
            "sample_results": sample_results,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        print(f"\n{'='*60}")
        print(f"[批量肿瘤定位] 处理完成")
        print(f"总图像数: {total_images}")
        print(f"检测到肿瘤: {images_with_tumor} ({detection_rate:.1f}%)")
        print(f"未检测到肿瘤: {images_without_tumor}")
        print(f"{'='*60}\n")
        
        return report
        
    except Exception as e:
        error_msg = f"批量处理发生错误: {str(e)}"
        print(f"[批量肿瘤定位] 错误: {error_msg}")
        return {
            "success": False,
            "error_message": error_msg,
            "input_directory": input_dir
        }


@tool
def get_localization_status() -> Dict:
    """
    获取肿瘤定位工具状态信息
    
    功能说明：
    - 检查模型是否已加载
    - 返回工具版本和配置信息
    - 诊断工具使用问题
    
    输出说明：
    返回包含以下信息的字典：
    - model_loaded: 模型是否已加载
    - model_path: 当前模型路径
    - device: 使用的设备（CPU或CUDA）
    - tool_version: 工具版本
    - supported_formats: 支持的图像格式
    
    使用示例：
    - get_localization_status()
    """
    
    global _localization_model, _model_path, _device
    
    default_model_path = os.path.join(
        os.path.dirname(__file__),
        "tool",
        "Tumor_Localization",
        "checkpoint_epoch_last.pth"
    )
    
    return {
        "model_loaded": _localization_model is not None,
        "model_path": _model_path,
        "default_model_path": default_model_path,
        "model_exists": os.path.exists(default_model_path),
        "device": str(_device) if _device else "未初始化",
        "tool_version": "1.0.0",
        "supported_formats": ["png", "jpg", "jpeg"],
        "default_threshold": 0.5,
        "model_architecture": "U-Net with SE attention and ASPP",
        "description": "基于U-Net的CT图像肿瘤分割和定位工具",
        "functions": {
            "tumor_localization_tool": "单张图像肿瘤分割和定位",
            "batch_tumor_localization": "批量图像肿瘤分割",
            "get_localization_status": "获取工具状态信息"
        }
    }


# ==============================================================================
# 工具注册函数
# ==============================================================================

def list_tumor_localization_tools():
    """
    返回肿瘤定位工具列表
    
    Returns:
        list: 包含所有肿瘤定位工具的列表
    """
    return [
        tumor_localization_tool,
        batch_tumor_localization,
        get_localization_status
    ]


def get_tumor_localization_tools():
    """
    获取肿瘤定位工具实例列表（用于智能体工具注册）
    
    Returns:
        list: LangChain工具实例列表
    """
    return [
        tumor_localization_tool,
        batch_tumor_localization,
        get_localization_status
    ]


# ==============================================================================
# 测试函数
# ==============================================================================

if __name__ == "__main__":
    print("肿瘤定位工具模块")
    print("=" * 60)
    
    # 测试模型加载
    print("\n测试模型加载...")
    model = get_localization_model()
    
    if model:
        print("✓ 模型加载成功")
        
        # 测试状态查询
        status = get_localization_status()
        print("\n工具状态:")
        for key, value in status.items():
            print(f"  {key}: {value}")
    else:
        print("✗ 模型加载失败")
