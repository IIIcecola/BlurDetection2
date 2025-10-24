import sys
import os
import argparse
import logging
import pathlib
import json
import shutil
import traceback
import subprocess
import tempfile
from datetime import datetime

import cv2
import numpy as np

from blur_detection.detection import estimate_blur, fix_image_size, pretty_blur_map
from blur_detection.blur_detection import BlurDetect

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 支持的媒体文件扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}


def parse_args():
    """解析命令行参数，新增局部模糊和I帧相关参数"""
    parser = argparse.ArgumentParser(description='增强版模糊检测工具（支持I帧提取和局部模糊过滤）')
    # 基础参数
    parser.add_argument('-i', '--inputs', type=str, nargs='+', required=True, 
                      help='输入路径（支持单个文件、多级目录）')
    parser.add_argument('-o', '--output-root', type=str, default='output', 
                      help='输出根目录（默认：output）')
    parser.add_argument('-t', '--threshold', type=float, default=100.0, 
                      help='模糊评分阈值（默认：100.0，低于此值的区域视为局部模糊）')
    parser.add_argument('-f', '--variable-size', action='store_true', 
                      help='不固定图像尺寸（默认：固定为200万像素）')
    
    # 局部模糊区域过滤参数
    parser.add_argument('-a', '--area-ratio', type=float, default=0.3, 
                      help='模糊区域占比阈值（默认：0.3，超过此比例判定为模糊）')
    
    # 视频I帧检测参数
    parser.add_argument('-r', '--blur-iframe-ratio', type=float, default=0.3, 
                      help='视频模糊I帧比例阈值（默认：0.3，超过此比例判定视频模糊）')
    
    # 结果保存控制参数
    parser.add_argument('--no-save-json', action='store_true', 
                      help='不保存检测结果JSON（默认：保存）')
    parser.add_argument('--no-save-blurmap', action='store_true', 
                      help='不保存模糊映射图（默认：保存）')
    
    return parser.parse_args()


def find_media_files(input_paths):
    """递归查找所有图片和视频文件（支持多级目录）"""
    media_files = []  # 元素格式：(文件路径, 类型('image'/'video'))
    for path_str in input_paths:
        path = pathlib.Path(path_str)
        if not path.exists():
            logger.warning(f"路径不存在：{path}，已跳过")
            continue
        
        if path.is_file():
            ext = path.suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                media_files.append((path, 'image'))
            elif ext in VIDEO_EXTENSIONS:
                media_files.append((path, 'video'))
            else:
                logger.info(f"不支持的文件类型：{path}，已跳过")
        elif path.is_dir():
            # 递归扫描多级目录
            for root, _, files in os.walk(path):
                for file in files:
                    file_path = pathlib.Path(root) / file
                    ext = file_path.suffix.lower()
                    if ext in IMAGE_EXTENSIONS:
                        media_files.append((file_path, 'image'))
                    elif ext in VIDEO_EXTENSIONS:
                        media_files.append((file_path, 'video'))
    return media_files


def get_output_paths(input_path, input_root, output_root):
    """计算输出路径（复刻原目录结构）"""
    relative_path = input_path.relative_to(input_root)
    output_file_path = pathlib.Path(output_root) / relative_path
    output_doc_dir = output_file_path.parent  # 检测文档与文件同目录
    return output_file_path, output_doc_dir


def save_blur_map(blur_map, save_path):
    """保存模糊映射图（统一处理归一化和保存逻辑）"""
    try:
        pretty_map = pretty_blur_map(blur_map)
        # 归一化到0-255以便显示
        pretty_map = ((pretty_map - pretty_map.min()) / 
                     (pretty_map.max() - pretty_map.min()) * 255).astype(np.uint8)
        cv2.imwrite(str(save_path), pretty_map)
        logger.debug(f"模糊映射图已保存至：{save_path}")
    except Exception as e:
        logger.error(f"保存模糊映射图失败：{str(e)}\n{traceback.format_exc()}")


def is_blurry_by_area(blur_map, threshold, area_ratio):
    """基于模糊区域占比判断是否模糊（核心局部过滤逻辑）"""
    total_pixels = blur_map.size
    if total_pixels == 0:
        return True, 1.0  # 空图像视为模糊
    threshold = float(threshold)
    # 统计模糊像素（转换为Python int）
    blurry_pixels = int(np.sum(blur_map < threshold))
    # 计算占比并转换为Python float
    blurry_ratio = float(blurry_pixels / total_pixels)
    # 返回Python原生bool和float
    return (blurry_ratio > area_ratio), blurry_ratio


def process_image(image_path, threshold, area_ratio, fix_size, 
                 output_file_path, output_doc_dir, 
                 save_json=True, save_blurmap=True):
    """处理单张图片（结合局部模糊区域过滤）"""
    try:
        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("无法读取图片文件")
        
        # 预处理（固定尺寸保证评分一致性）
        if fix_size:
            image = fix_image_size(image)
        
        # 模糊检测（获取模糊映射图和全局评分）
        blur_map, global_score, _ = estimate_blur(image, threshold=threshold)
        # 基于局部区域占比判断是否模糊
        is_blurry, blurry_ratio = is_blurry_by_area(blur_map, threshold, area_ratio)
        
        # 构造结果字典
        result = {
            'type': 'image',
            'input_path': str(image_path),
            'global_score': float(global_score),  # 全局模糊评分
            'blurry_area_ratio': float(blurry_ratio),  # 模糊区域占比
            'blurry': is_blurry,  # 是否整体模糊（基于区域占比）
            'threshold_used': threshold,
            'area_ratio_used': area_ratio,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存模糊映射图
        if save_blurmap:
            blurmap_path = output_doc_dir / f"{image_path.stem}_blurmap.png"
            save_blur_map(blur_map, blurmap_path)
            result['blurmap_path'] = str(blurmap_path)
        
        # 若不模糊，移动文件到输出目录（复刻结构）
        if not is_blurry:
            output_doc_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(image_path), str(output_file_path))
            logger.info(f"图片已移动至：{output_file_path}")
        
        # 保存JSON结果
        if save_json:
            json_path = output_doc_dir / f"{image_path.stem}_result.json"
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=4)
            logger.debug(f"图片检测结果已保存至：{json_path}")
        
        return result
    
    except Exception as e:
        logger.error(f"处理图片 {image_path} 失败：{str(e)}\n{traceback.format_exc()}")
        return None


def extract_i_frames(video_path, output_dir):
    """用ffmpeg提取视频中的I帧（关键帧）"""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ffmpeg命令：筛选I帧（pict_type=I）并输出为图片
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",  # 隐藏冗余输出
        "-i", str(video_path),
        "-vf", "select=eq(pict_type\\,I)",  # 筛选I帧
        "-vsync", "vfr",  # 去除重复帧
        f"{output_dir}/i_frame_%04d.jpg"  # 输出文件名格式（按序号命名）
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg提取I帧失败：{e.stderr}")
    
    # 获取所有提取的I帧路径（按序号排序）
    i_frames = sorted(output_dir.glob("i_frame_*.jpg"), key=lambda x: int(x.stem.split("_")[-1]))
    if not i_frames:
        raise ValueError(f"视频 {video_path} 中未检测到I帧（可能不是标准编码格式）")
    
    return i_frames


def process_video(video_path, threshold, area_ratio, blur_iframe_ratio,
                 fix_size, output_file_path, output_doc_dir,
                 save_json=True, save_blurmap=True):
    """处理单个视频（基于I帧检测和局部模糊过滤）"""
    try:
        # 临时目录保存提取的I帧（自动清理）
        with tempfile.TemporaryDirectory() as tmpdir:
            i_frames = extract_i_frames(video_path, tmpdir)
            logger.info(f"从视频 {video_path} 中提取到 {len(i_frames)} 个I帧")
            
            # 处理每个I帧
            iframe_results = []
            for idx, frame_path in enumerate(i_frames):
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    logger.warning(f"跳过损坏的I帧：{frame_path}")
                    continue
                
                # 预处理
                if fix_size:
                    frame = fix_image_size(frame)
                
                # 模糊检测（基于局部区域）
                blur_map, global_score, _ = estimate_blur(frame, threshold=threshold)
                is_blurry, blurry_ratio = is_blurry_by_area(blur_map, threshold, area_ratio)
                
                # 记录单I帧结果
                iframe_result = {
                    'iframe_idx': idx,  # I帧序号（0开始）
                    'global_score': float(global_score),
                    'blurry_area_ratio': float(blurry_ratio),
                    'blurry': is_blurry
                }
                
                # 保存I帧模糊映射图
                if save_blurmap:
                    blurmap_path = output_doc_dir / f"{video_path.stem}_iframe{idx}_blurmap.png"
                    save_blur_map(blur_map, blurmap_path)
                    iframe_result['blurmap_path'] = str(blurmap_path)
                
                iframe_results.append(iframe_result)
            
            # 计算模糊I帧比例，判定视频是否模糊
            total_iframe = len(iframe_results)
            if total_iframe == 0:
                raise ValueError("所有I帧处理失败，无法判定视频模糊性")
            
            blur_iframe_count = sum(1 for res in iframe_results if res['blurry'])
            blur_ratio = blur_iframe_count / total_iframe
            video_blurry = blur_ratio > blur_iframe_ratio
            
            # 构造视频总结果
            result = {
                'type': 'video',
                'input_path': str(video_path),
                'total_i_frames': total_iframe,
                'blur_i_frame_count': blur_iframe_count,
                'blur_i_frame_ratio': float(blur_ratio),
                'blurry': video_blurry,  # 视频是否模糊（基于I帧比例）
                'threshold_used': threshold,
                'area_ratio_used': area_ratio,
                'blur_iframe_ratio_used': blur_iframe_ratio,
                'i_frames': iframe_results,  # 所有I帧的详细结果
                'timestamp': datetime.now().isoformat()
            }
            
            # 若不模糊，移动视频到输出目录（复刻结构）
            if not video_blurry:
                output_doc_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(video_path), str(output_file_path))
                logger.info(f"视频已移动至：{output_file_path}")
            
            # 保存JSON结果
            if save_json:
                json_path = output_doc_dir / f"{video_path.stem}_result.json"
                with open(json_path, 'w') as f:
                    json.dump(result, f, indent=4)
                logger.debug(f"视频检测结果已保存至：{json_path}")
            
            return result
    
    except Exception as e:
        logger.error(f"处理视频 {video_path} 失败：{str(e)}\n{traceback.format_exc()}")
        return None


def main():
    args = parse_args()
    input_paths = [pathlib.Path(p).resolve() for p in args.inputs]
    output_root = pathlib.Path(args.output_root).resolve()
    fix_size = not args.variable_size  # 是否固定图像尺寸（默认固定）
    
    # 确保输出根目录存在
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 递归查找所有媒体文件（图片+视频，支持多级目录）
    media_files = find_media_files(args.inputs)
    logger.info(f"共发现 {len(media_files)} 个媒体文件待处理")
    
    # 记录所有处理结果（用于汇总）
    all_results = []
    
    for media_path, media_type in media_files:
        logger.info(f"开始处理：{media_path}（类型：{media_type}）")
        
        # 确定输入根目录（用于复刻目录结构）
        input_root = None
        for root in input_paths:
            if media_path.is_relative_to(root):
                input_root = root
                break
        if input_root is None:
            input_root = media_path.parent if media_path.is_file() else media_path
        
        # 计算输出路径（复刻原目录结构）
        output_file_path, output_doc_dir = get_output_paths(
            media_path, input_root, output_root
        )
        
        # 处理媒体文件（区分图片/视频）
        if media_type == 'image':
            result = process_image(
                image_path=media_path,
                threshold=args.threshold,
                area_ratio=args.area_ratio,
                fix_size=fix_size,
                output_file_path=output_file_path,
                output_doc_dir=output_doc_dir,
                save_json=not args.no_save_json,
                save_blurmap=not args.no_save_blurmap
            )
        else:  # 视频
            result = process_video(
                video_path=media_path,
                threshold=args.threshold,
                area_ratio=args.area_ratio,
                blur_iframe_ratio=args.blur_iframe_ratio,
                fix_size=fix_size,
                output_file_path=output_file_path,
                output_doc_dir=output_doc_dir,
                save_json=not args.no_save_json,
                save_blurmap=not args.no_save_blurmap
            )
        
        if result:
            all_results.append(result)
    
    # 保存总结果汇总JSON
    summary_json = output_root / "detection_summary.json"
    with open(summary_json, 'w') as f:
        json.dump({
            'total_files': len(media_files),
            'processed_files': len(all_results),
            'timestamp': datetime.now().isoformat(),
            'parameters_used': {
                'threshold': args.threshold,
                'area_ratio': args.area_ratio,
                'blur_iframe_ratio': args.blur_iframe_ratio,
                'fix_size': not args.variable_size
            },
            'results': all_results
        }, f, indent=4)
    logger.info(f"所有检测结果汇总已保存至：{summary_json}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.critical(f"程序运行失败：{str(e)}\n{traceback.format_exc()}")
        sys.exit(1)
