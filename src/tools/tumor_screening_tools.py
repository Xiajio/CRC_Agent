"""
肿瘤识别筛选工具模块

提供基于YOLO模型的CT图像肿瘤检测和筛选功能。
用于在大量CT图像中自动筛选出包含肿瘤信息的图像样本。
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

# 注意：ultralytics 和 cv2 使用延迟导入
# 这样即使这些包没安装，智能体也能启动，只是不能使用肿瘤筛选功能

# 尝试加载模型（延迟初始化）
_tumor_model = None
_model_path = None


def _import_ultralytics():
    """延迟导入 ultralytics 模块"""
    global _ultralytics_imported
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        return None


def _import_cv2():
    """延迟导入 cv2 模块"""
    try:
        import cv2
        return cv2
    except ImportError:
        return None


def get_tumor_model(model_path: str = None):
    """
    获取或初始化YOLO肿瘤检测模型（单例模式，懒加载）
    
    Args:
        model_path: 模型文件路径，如果为None则使用默认路径
        
    Returns:
        YOLO模型实例，如果导入失败则返回None
    """
    global _tumor_model, _model_path
    
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(__file__),
            "tool",
            "Tumor_Detection",
            "best.pt"
        )
    
    # 如果模型已加载且路径相同，直接返回
    if _tumor_model is not None and _model_path == model_path:
        return _tumor_model
    
    # 延迟导入 ultralytics
    YOLO = _import_ultralytics()
    if YOLO is None:
        print("[TumorScreening] 警告: ultralytics 模块未安装，无法加载模型")
        print("[TumorScreening] 请运行: pip install ultralytics")
        return None
    
    try:
        # 检查CUDA可用性并设置设备
        import torch
        if torch.cuda.is_available():
            use_gpu = True
            print(f"[TumorScreening] 检测到CUDA GPU: {torch.cuda.get_device_name(0)}")
            print(f"[TumorScreening] GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            use_gpu = False
            print("[TumorScreening] 未检测到CUDA，使用CPU模式")
        
        print(f"[TumorScreening] 正在加载YOLO模型: {model_path}")
        _tumor_model = YOLO(model_path)
        
        # 如果使用GPU，将模型移动到CUDA设备
        if use_gpu:
            _tumor_model = _tumor_model.cuda()
        
        _model_path = model_path
        print(f"[TumorScreening] 模型加载成功 (设备: {'GPU' if use_gpu else 'CPU'})")
        return _tumor_model
    except Exception as e:
        print(f"[TumorScreening] 模型加载失败: {e}")
        return None


def reset_tumor_model():
    """重置模型（用于测试或重新初始化）"""
    global _tumor_model, _model_path
    _tumor_model = None
    _model_path = None


# ==============================================================================
# LangChain Tool 定义
# ==============================================================================

@tool
def tumor_screening_tool(
    input_dir: str,
    output_dir: str = None,
    model_path: str = None,
    confidence_threshold: float = 0.5,
    image_extensions: List[str] = None
) -> Dict:
    """
    肿瘤图像筛选工具 - 使用YOLO模型在CT图像目录中筛选包含肿瘤的图像
    
    功能说明：
    - 遍历指定目录下的所有图像
    - 使用训练好的YOLO模型进行肿瘤检测
    - 将检测到肿瘤的图像复制到输出目录
    - 支持保持原始文件夹结构
    - 返回详细的筛选报告
    
    输入参数：
    - input_dir: 输入图像目录路径（包含CT图像的文件夹）
    - output_dir: 输出目录路径（可选，默认在输入目录旁创建筛选结果目录）
    - model_path: YOLO模型文件路径（可选，使用默认模型）
    - confidence_threshold: 检测置信度阈值，0-1之间，值越高越严格（默认0.5）
    - image_extensions: 图像文件扩展名列表（可选，默认['.png', '.jpg', '.jpeg']）
    
    输出说明：
    返回包含以下信息的字典：
    - success: 是否成功完成筛选
    - total_images: 处理的总图像数量
    - images_with_tumor: 检测到肿瘤的图像数量
    - images_without_tumor: 未检测到肿瘤的图像数量
    - tumor_ratio: 肿瘤图像占比
    - output_dir: 输出目录路径
    - sample_images: 包含肿瘤的图像样本列表（最多10个）
    -筛查时间: 筛选完成的时间戳
    - error_message: 如果失败，包含错误信息
    
    使用示例：
    - tumor_screening_tool(input_dir="/path/to/ct_images")
    - tumor_screening_tool(input_dir="/data/CT", output_dir="/results/tumor_images", confidence_threshold=0.6)
    
    注意事项：
    - 模型文件(best.pt)需要预先训练并放置在正确位置
    - 输入目录不能包含中文字符（可能导致路径读取问题）
    - 建议置信度阈值设置在0.5-0.7之间，平衡精确率和召回率
    """
    
    # 参数验证和默认值设置
    if not input_dir or not input_dir.strip():
        return {
            "success": False,
            "error_message": "错误：必须提供输入目录路径 (input_dir)",
            "example_usage": "tumor_screening_tool(input_dir='/path/to/ct_images', confidence_threshold=0.5)"
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
            f"{base_name}_tumor_screened"
        )
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        return {
            "success": False,
            "error_message": f"错误：输入目录不存在: {input_dir}",
            "suggestion": "请检查路径是否正确，或提供完整的目录路径"
        }
    
    # 获取模型
    model = get_tumor_model(model_path)
    if model is None:
        return {
            "success": False,
            "error_message": "错误：无法加载肿瘤检测模型",
            "model_path": model_path or os.path.join(os.path.dirname(__file__), "Tumor_Detection", "best.pt"),
            "suggestion": "请确保模型文件(best.pt)存在于正确的目录中"
        }
    
    # 初始化计数器
    total_images = 0
    images_with_tumor = 0
    images_without_tumor = 0
    sample_images = []
    errors = []
    
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"[肿瘤筛选工具] 开始处理")
        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")
        print(f"置信度阈值: {confidence_threshold}")
        print(f"{'='*60}\n")

        # 先统计所有图片数量用于计算进度
        all_image_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                file_lower = file.lower()
                if any(file_lower.endswith(ext) for ext in image_extensions):
                    all_image_files.append(os.path.join(root, file))

        total_to_process = len(all_image_files)
        print(f"[PROGRESS] total_images={total_to_process}")

        if total_to_process == 0:
            return {
                "success": False,
                "error_message": f"在目录 {input_dir} 中未找到支持的图像文件",
                "supported_formats": image_extensions
            }

        # 遍历输入目录进行筛选
        for idx, input_image_path in enumerate(all_image_files):
            file = os.path.basename(input_image_path)
            total_images += 1

            # 计算相对路径以保持文件夹结构
            relative_path = os.path.relpath(os.path.dirname(input_image_path), input_dir)
            output_image_dir = os.path.join(output_dir, relative_path)
            output_image_path = os.path.join(output_image_dir, file)

            # 确保输出子目录存在
            os.makedirs(output_image_dir, exist_ok=True)

            # 读取图像
            try:
                cv2 = _import_cv2()
                if cv2 is None:
                    raise ImportError("cv2 module not available")
                image = cv2.imread(input_image_path)
                if image is None:
                    errors.append(f"无法读取图像: {input_image_path}")
                    images_without_tumor += 1
                    # 输出进度
                    progress = (idx + 1) / total_to_process
                    print(f"[PROGRESS] {idx+1}/{total_to_process} - {progress:.2%} - {file}")
                    continue
            except ImportError:
                errors.append("cv2模块未安装，无法读取图像")
                return {
                    "success": False,
                    "error_message": "cv2模块未安装",
                    "suggestion": "请运行: pip install opencv-python"
                }
            except Exception as e:
                errors.append(f"读取图像失败 {input_image_path}: {str(e)}")
                images_without_tumor += 1
                # 输出进度
                progress = (idx + 1) / total_to_process
                print(f"[PROGRESS] {idx+1}/{total_to_process} - {progress:.2%} - {file}")
                continue

            # 运行模型预测
            try:
                results = model.predict(source=image, verbose=False)
            except Exception as e:
                errors.append(f"模型预测失败 {input_image_path}: {str(e)}")
                images_without_tumor += 1
                # 输出进度
                progress = (idx + 1) / total_to_process
                print(f"[PROGRESS] {idx+1}/{total_to_process} - {progress:.2%} - {file}")
                continue

            # 检查检测结果
            has_tumor = False
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        confidence = box.conf.item()
                        if confidence >= confidence_threshold:
                            has_tumor = True
                            break
                    if has_tumor:
                        break

            if has_tumor:
                # 复制图像到输出目录
                try:
                    shutil.copy2(input_image_path, output_image_path)
                    images_with_tumor += 1

                    # 保存样本路径（最多10个）
                    if len(sample_images) < 10:
                        sample_images.append({
                            "path": output_image_path,
                            "relative_path": os.path.join(relative_path, file),
                            "confidence": confidence
                        })

                    print(f"✓ 检测到肿瘤: {file} (置信度: {confidence:.2f})")
                except Exception as e:
                    errors.append(f"复制图像失败 {input_image_path}: {str(e)}")
            else:
                images_without_tumor += 1
                print(f"✗ 无肿瘤: {file}")

            # 输出进度
            progress = (idx + 1) / total_to_process
            print(f"[PROGRESS] {idx+1}/{total_to_process} - {progress:.2%} - {file}")
        
        # 计算肿瘤占比
        tumor_ratio = (images_with_tumor / total_images * 100) if total_images > 0 else 0
        
        # 生成筛选报告
        screening_report = {
            "success": True,
            "summary": {
                "total_images_processed": total_images,
                "images_with_tumor": images_with_tumor,
                "images_without_tumor": images_without_tumor,
                "tumor_detection_rate": f"{tumor_ratio:.1f}%",
                "confidence_threshold_used": confidence_threshold
            },
            "output_directory": output_dir,
            "sample_positive_images": sample_images[:10],
            "screening_timestamp": datetime.now().isoformat(),
            "statistics": {
                "model_path": _model_path,
                "input_directory": input_dir,
                "processing_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # 添加错误摘要（如果有）
        if errors:
            screening_report["errors"] = {
                "total_errors": len(errors),
                "error_samples": errors[:5]  # 只保留前5个错误示例
            }
        
        # 打印总结
        print(f"\n{'='*60}")
        print(f"[肿瘤筛选工具] 筛选完成")
        print(f"总图像数: {total_images}")
        print(f"检测到肿瘤: {images_with_tumor} ({tumor_ratio:.1f}%)")
        print(f"未检测到肿瘤: {images_without_tumor}")
        print(f"输出目录: {output_dir}")
        print(f"{'='*60}\n")
        
        return screening_report
        
    except Exception as e:
        error_msg = f"筛选过程发生错误: {str(e)}"
        print(f"[肿瘤筛选工具] 错误: {error_msg}")
        return {
            "success": False,
            "error_message": error_msg,
            "input_directory": input_dir,
            "output_directory": output_dir,
            "suggestion": "请检查输入目录权限和路径格式，避免使用中文路径"
        }


@tool
def quick_tumor_check(image_path: str, model_path: str = None) -> Dict:
    """
    快速肿瘤检测工具 - 检测单张图像是否包含肿瘤
    
    功能说明：
    - 对单张CT图像进行快速肿瘤检测
    - 返回检测结果和置信度
    - 适用于随机抽查或验证场景
    
    输入参数：
    - image_path: 待检测图像的完整路径
    - model_path: YOLO模型文件路径（可选）
    
    输出说明：
    返回包含以下信息的字典：
    - has_tumor: 是否检测到肿瘤 (true/false)
    - confidence: 最高检测置信度
    - bounding_boxes: 检测框坐标列表
    - processing_time: 处理时间信息
    - error_message: 如果失败，包含错误信息
    
    使用示例：
    - quick_tumor_check(image_path="/path/to/ct_image.png")
    - quick_tumor_check(image_path="/data/image.jpg", model_path="/custom/path/best.pt")
    """
    
    # 验证输入
    if not image_path or not image_path.strip():
        return {
            "success": False,
            "error_message": "必须提供图像路径 (image_path)"
        }
    
    image_path = image_path.strip()
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        return {
            "success": False,
            "error_message": f"图像文件不存在: {image_path}",
            "suggestion": "请检查路径是否正确"
        }
    
    # 获取模型
    model = get_tumor_model(model_path)
    if model is None:
        return {
            "success": False,
            "error_message": "无法加载肿瘤检测模型"
        }
    
    try:
        # 读取图像
        cv2 = _import_cv2()
        if cv2 is None:
            return {
                "success": False,
                "error_message": "cv2模块未安装",
                "suggestion": "请运行: pip install opencv-python"
            }
        image = cv2.imread(image_path)
        if image is None:
            return {
                "success": False,
                "error_message": f"无法读取图像: {image_path}"
            }
        
        # 运行预测
        results = model.predict(source=image, verbose=False)
        
        # 分析结果
        has_tumor = False
        max_confidence = 0.0
        bounding_boxes = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    confidence = box.conf.item()
                    if confidence > max_confidence:
                        max_confidence = confidence
                    
                    # 提取边界框坐标
                    if confidence >= 0.5:  # 使用默认阈值
                        has_tumor = True
                        bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                        bounding_boxes.append({
                            "x1": int(bbox[0]),
                            "y1": int(bbox[1]),
                            "x2": int(bbox[2]),
                            "y2": int(bbox[3]),
                            "confidence": round(confidence, 4)
                        })
        
        return {
            "success": True,
            "has_tumor": has_tumor,
            "confidence": round(max_confidence, 4) if max_confidence > 0 else None,
            "bounding_boxes": bounding_boxes,
            "total_detections": len(bounding_boxes),
            "image_path": os.path.basename(image_path),
            "processing_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error_message": f"检测失败: {str(e)}",
            "image_path": image_path
        }


@tool
def get_tumor_screening_status() -> Dict:
    """
    获取肿瘤筛选工具状态信息
    
    功能说明：
    - 检查模型是否已加载
    - 返回工具版本和配置信息
    - 诊断工具使用问题
    
    输出说明：
    返回包含以下信息的字典：
    - model_loaded: 模型是否已加载
    - model_path: 当前模型路径
    - default_model_path: 默认模型路径
    - tool_version: 工具版本
    - supported_formats: 支持的图像格式
    
    使用示例：
    - get_tumor_screening_status()
    """
    
    global _tumor_model, _model_path
    
    default_model_path = os.path.join(
        os.path.dirname(__file__),
        "Tumor_Detection",
        "best.pt"
    )
    
    return {
        "model_loaded": _tumor_model is not None,
        "model_path": _model_path,
        "default_model_path": default_model_path,
        "model_exists": os.path.exists(default_model_path),
        "tool_version": "1.0.0",
        "supported_formats": ["png", "jpg", "jpeg"],
        "default_confidence_threshold": 0.5,
        "description": "基于YOLOv8的CT图像肿瘤检测和筛选工具",
        "functions": {
            "tumor_screening_tool": "批量筛选包含肿瘤的CT图像",
            "quick_tumor_check": "单张图像快速肿瘤检测",
            "get_tumor_screening_status": "获取工具状态信息"
        }
    }


def perform_comprehensive_tumor_check(patient_id: str) -> Dict:
    """
    高级聚合工具：输入患者ID，自动执行完整的肿瘤检测流程
    
    功能说明：
    - 输入患者ID（如 "93" 或 "093"）
    - 自动查找对应的影像文件夹
    - 遍历所有CT切片进行肿瘤检测
    - 返回汇总的检测报告
    
    输入参数：
    - patient_id: 患者编号，可以是数字字符串（如 "93"）或整数
    
    输出说明：
    返回包含以下信息的字典：
    - patient_id: 患者ID
    - total_images: 总影像数量
    - images_with_tumor: 检测到肿瘤的影像数量
    - images_without_tumor: 未检测到肿瘤的影像数量
    - tumor_detection_rate: 肿瘤检出率
    - has_tumor: 是否检测到肿瘤（任一影像检出即为True）
    - all_results: 所有影像的详细检测结果
    - sample_images_with_tumor: 包含肿瘤的影像样本（最多5个）
    - processing_timestamp: 处理时间戳
    - error: 如果失败，包含错误信息
    
    使用示例：
    - perform_comprehensive_tumor_check(patient_id="93")
    - perform_comprehensive_tumor_check(patient_id="001")
    
    适用场景：
    - 肿瘤检测/癌症筛查请求
    - 病灶识别分析
    - 影像学AI辅助诊断
    """
    
    # 参数验证
    if not patient_id:
        return {
            "error": "必须提供患者ID (patient_id)",
            "has_tumor": False
        }
    
    # 处理患者ID格式
    patient_id_str = str(patient_id).strip()
    
    # 转换为3位数字格式（自动补零）
    if patient_id_str.isdigit():
        folder_name = patient_id_str.zfill(3)
    else:
        return {
            "error": f"无效的患者ID格式: {patient_id}，请提供数字格式的ID",
            "has_tumor": False
        }
    
    # 构建影像文件夹路径
    base_path = Path(r"E:/LangG/data/Case Database/Radiographic Imaging")
    target_folder = base_path / folder_name
    
    # 检查文件夹是否存在
    if not target_folder.exists():
        return {
            "error": f"未找到患者 {folder_name} 的影像目录: {target_folder}",
            "has_tumor": False,
            "patient_id": folder_name
        }
    
    # 查找所有影像文件（支持多种格式）
    images = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        images.extend(sorted(target_folder.glob(ext)))
    
    if not images:
        return {
            "error": f"患者 {folder_name} 的影像目录为空，未找到影像文件",
            "has_tumor": False,
            "patient_id": folder_name
        }
    
    print(f"[TumorScreening] 开始处理患者 {folder_name} 的 {len(images)} 张影像...")
    
    # 遍历所有影像进行检测
    all_results = []
    images_with_tumor = 0
    images_without_tumor = 0
    sample_images_with_tumor = []
    max_confidence = 0.0
    total_detections = 0
    
    for idx, image_path in enumerate(images):
        # 调用快速肿瘤检测工具
        result = quick_tumor_check.invoke({"image_path": str(image_path)})
        
        if isinstance(result, dict):
            # 记录结果
            all_results.append({
                "image_name": image_path.name,
                "image_path": str(image_path),
                "has_tumor": result.get("has_tumor", False),
                "confidence": result.get("confidence"),
                "total_detections": result.get("total_detections", 0),
                "bounding_boxes": result.get("bounding_boxes", [])
            })
            
            if result.get("has_tumor", False):
                images_with_tumor += 1
                total_detections += result.get("total_detections", 0)
                if result.get("confidence") and result.get("confidence") > max_confidence:
                    max_confidence = result.get("confidence")
                # 保存样本（最多5个）
                if len(sample_images_with_tumor) < 5:
                    sample_images_with_tumor.append({
                        "image_name": image_path.name,
                        "image_path": str(image_path),
                        "confidence": result.get("confidence"),
                        "total_detections": result.get("total_detections", 0)
                    })
                print(f"  [{idx+1}/{len(images)}] ✓ {image_path.name} - 检测到肿瘤")
            else:
                images_without_tumor += 1
                print(f"  [{idx+1}/{len(images)}] ✗ {image_path.name} - 无肿瘤")
        else:
            # 检测失败
            all_results.append({
                "image_name": image_path.name,
                "image_path": str(image_path),
                "has_tumor": False,
                "error": str(result)
            })
            images_without_tumor += 1
            print(f"  [{idx+1}/{len(images)}] ⚠ {image_path.name} - 检测失败")
    
    # 计算检出率
    tumor_detection_rate = (images_with_tumor / len(images) * 100) if len(images) > 0 else 0
    
    # 构建汇总结果
    summary = {
        "success": True,
        "patient_id": folder_name,
        "total_images": len(images),
        "images_with_tumor": images_with_tumor,
        "images_without_tumor": images_without_tumor,
        "tumor_detection_rate": f"{tumor_detection_rate:.1f}%",
        "has_tumor": images_with_tumor > 0,
        "max_confidence": round(max_confidence, 4) if max_confidence > 0 else None,
        "total_detections": total_detections,
        "all_results": all_results,
        "sample_images_with_tumor": sample_images_with_tumor,
        "processing_timestamp": datetime.now().isoformat()
    }
    
    print(f"[TumorScreening] 患者 {folder_name} 检测完成: {images_with_tumor}/{len(images)} 张检测到肿瘤")
    
    return summary


# ==============================================================================
# 工具注册函数
# ==============================================================================

def list_tumor_screening_tools():
    """
    返回肿瘤筛选工具列表
    
    Returns:
        list: 包含所有肿瘤筛选工具的列表
    """
    return [
        tumor_screening_tool,
        quick_tumor_check,
        get_tumor_screening_status,
        perform_comprehensive_tumor_check
    ]


def get_tumor_screening_tools():
    """
    获取肿瘤筛选工具实例列表（用于智能体工具注册）
    
    Returns:
        list: LangChain工具实例列表
    """
    return [
        tumor_screening_tool,
        quick_tumor_check,
        get_tumor_screening_status,
        perform_comprehensive_tumor_check_as_tool  # 使用包装后的工具
    ]


# ==============================================================================
# 使用 StructuredTool 创建 perform_comprehensive_tumor_check 工具
# ==============================================================================

class PerformTumorCheckInput(BaseModel):
    """perform_comprehensive_tumor_check 的输入参数"""
    patient_id: str = Field(
        ...,
        description="患者编号，可以是数字字符串（如 '93'）或整数"
    )

def _execute_tumor_check(patient_id: str) -> Dict:
    """
    内部执行函数：输入患者ID，自动执行完整的肿瘤检测流程
    """
    return perform_comprehensive_tumor_check(patient_id)

# 创建 LangChain StructuredTool
perform_comprehensive_tumor_check_as_tool = StructuredTool(
    func=_execute_tumor_check,
    name="perform_comprehensive_tumor_check",
    description="高级聚合工具：输入患者ID，自动执行完整的肿瘤检测流程。触发条件：用户想对患者影像进行肿瘤检测、癌症筛查、病灶识别。触发关键词：肿瘤检测、癌症筛查、影像诊断、病灶分析、CT检测",
    args_schema=PerformTumorCheckInput
)
