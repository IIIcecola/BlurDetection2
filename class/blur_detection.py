import sys
import os
import logging
import pathlib
import json
import shutil
import traceback
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from blur_detection import estimate_blur, fix_image_size, pretty_blur_map

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


class BlurDetector:
    def __init__(self, input_root_dir, output_root_dir, 
                 threshold=100.0, area_ratio=0.3, 
                 blur_iframe_ratio=0.3, variable_size=False,
                 save_json=True, save_blurmap=True):
        """
        初始化模糊检测工具
        
        参数:
            input_root_dir: 输入媒体文件根目录
            output_root_dir: 输出结果根目录
            threshold: 模糊评分阈值（默认：100.0，低于此值的区域视为局部模糊）
            area_ratio: 模糊区域占比阈值（默认：0.3，超过此比例判定为模糊）
            blur_iframe_ratio: 视频模糊I帧比例阈值（默认：0.3）
            variable_size: 是否不固定图像尺寸（默认：False，即固定为200万像素）
            save_json: 是否保存检测结果JSON（默认：True）
            save_blurmap: 是否保存模糊映射图（默认：True）
        """
        self.input_root = Path(input_root_dir).resolve()
        self.output_root = Path(output_root_dir).resolve()
        
        # 检测参数
        self.threshold = threshold
        self.area_ratio = area_ratio
        self.blur_iframe_ratio = blur_iframe_ratio
        self.fix_size = not variable_size  # 是否固定图像尺寸
        self.save_json = save_json
        self.save_blurmap = save_blurmap
        
        # 创建输出根目录
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # 初始化统计信息
        self.stats = {
            "total_processed": 0,
            "image_processed": 0,
            "video_processed": 0,
            "blurry_detected": 0,
            "non_blurry_kept": 0,
            "processing_errors": 0
        }
        
        # 存储处理结果
        self.process_results = []
        
        logger.info(f"初始化模糊检测工具:")
        logger.info(f"输入目录: {self.input_root}")
        logger.info(f"输出目录: {self.output_root}")
        logger.info(f"模糊评分阈值: {self.threshold}")
        logger.info(f"模糊区域占比阈值: {self.area_ratio}")
        logger.info(f"视频模糊I帧比例阈值: {self.blur_iframe_ratio}")
        logger.info(f"是否固定图像尺寸: {self.fix_size}")

    def check_environment(self):
      """检查并返回当前运行环境信息"""
      return {
          "python_path": sys.executable,
          "conda_env": os.environ.get('CONDA_DEFAULT_ENV', '未激活'),
      }
  
    def _get_output_paths(self, input_path):
        """计算输出路径（保持原始目录结构）"""
        try:
            # 获取相对于输入根目录的相对路径
            relative_path = input_path.relative_to(self.input_root)
            output_file_path = self.output_root / relative_path
            output_doc_dir = output_file_path.parent  # 输出目录
            return output_file_path, output_doc_dir
        except ValueError as e:
            logger.error(f"计算输出路径失败: {str(e)}")
            raise

    def _save_blur_map(self, blur_map, save_path):
        """保存模糊映射图（统一处理归一化和保存逻辑）"""
        try:
            pretty_map = pretty_blur_map(blur_map)
            # 归一化到0-255以便显示
            pretty_map = ((pretty_map - pretty_map.min()) / 
                         (pretty_map.max() - pretty_map.min()) * 255).astype(np.uint8)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), pretty_map)
            logger.debug(f"模糊映射图已保存至：{save_path}")
            return True
        except Exception as e:
            logger.error(f"保存模糊映射图失败：{str(e)}\n{traceback.format_exc()}")
            return False

    def _is_blurry_by_area(self, blur_map):
        """基于模糊区域占比判断是否模糊（核心局部过滤逻辑）"""
        try:
            total_pixels = blur_map.size
            if total_pixels == 0:
                return True, 1.0  # 空图像视为模糊
            
            # 统计模糊像素
            blurry_pixels = int(np.sum(blur_map < self.threshold))
            # 计算占比
            blurry_ratio = float(blurry_pixels / total_pixels)
            return (blurry_ratio > self.area_ratio), blurry_ratio
        except Exception as e:
            logger.error(f"判断模糊区域占比失败：{str(e)}\n{traceback.format_exc()}")
            return True, 1.0  # 出错时默认视为模糊

    def _process_image(self, image_path):
        """处理单张图片（内部方法）"""
        try:
            # 读取图片
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("无法读取图片文件")
            
            # 预处理（固定尺寸保证评分一致性）
            if self.fix_size:
                image = fix_image_size(image)
            
            # 模糊检测（获取模糊映射图和全局评分）
            blur_map, global_score, _ = estimate_blur(image, threshold=self.threshold)
            # 基于局部区域占比判断是否模糊
            is_blurry, blurry_ratio = self._is_blurry_by_area(blur_map)
            
            # 计算输出路径
            output_file_path, output_doc_dir = self._get_output_paths(image_path)
            
            # 构造结果字典
            result = {
                'type': 'image',
                'input_path': str(image_path),
                'output_path': str(output_file_path),
                'global_score': float(global_score),
                'blurry_area_ratio': float(blurry_ratio),
                'blurry': is_blurry,
                'threshold_used': self.threshold,
                'area_ratio_used': self.area_ratio,
                'timestamp': datetime.now().isoformat()
            }
            
            # 保存模糊映射图
            if self.save_blurmap:
                blurmap_path = output_doc_dir / f"{image_path.stem}_blurmap.png"
                if self._save_blur_map(blur_map, blurmap_path):
                    result['blurmap_path'] = str(blurmap_path)
            
            # 若不模糊，移动文件到输出目录（保持目录结构）
            if not is_blurry:
                output_doc_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(image_path), str(output_file_path))
                self.stats["non_blurry_kept"] += 1
                logger.info(f"非模糊图片已保存至：{output_file_path}")
            else:
                self.stats["blurry_detected"] += 1
                logger.info(f"检测到模糊图片：{image_path}")
            
            # 保存JSON结果
            if self.save_json:
                json_path = output_doc_dir / f"{image_path.stem}_blur_result.json"
                output_doc_dir.parent.mkdir(parents=True, exist_ok=True)
                with open(json_path, 'w') as f:
                    json.dump(result, f, indent=4)
                result['json_path'] = str(json_path)
            
            return result
        
        except Exception as e:
            error_msg = f"处理图片 {image_path} 失败：{str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.stats["processing_errors"] += 1
            return {
                'type': 'image',
                'input_path': str(image_path),
                'status': 'error',
                'error_message': error_msg,
                'timestamp': datetime.now().isoformat()
            }

    def _extract_i_frames(self, video_path, output_dir):
        """用ffmpeg提取视频中的I帧（内部方法）"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # ffmpeg命令：筛选I帧（pict_type=I）并输出为图片
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", str(video_path),
                "-vf", "select=eq(pict_type\\,I)",
                "-vsync", "vfr",
                f"{output_dir}/i_frame_%04d.jpg"
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # 获取所有提取的I帧路径（按序号排序）
            i_frames = sorted(output_dir.glob("i_frame_*.jpg"), 
                             key=lambda x: int(x.stem.split("_")[-1]))
            if not i_frames:
                raise ValueError(f"视频 {video_path} 中未检测到I帧")
            
            return i_frames
        except Exception as e:
            error_msg = f"提取I帧失败：{str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _process_video(self, video_path):
        """处理单个视频（内部方法）"""
        try:
            # 计算输出路径
            output_file_path, output_doc_dir = self._get_output_paths(video_path)
            
            # 临时目录保存提取的I帧（自动清理）
            with tempfile.TemporaryDirectory() as tmpdir:
                i_frames = self._extract_i_frames(video_path, tmpdir)
                logger.info(f"从视频 {video_path} 中提取到 {len(i_frames)} 个I帧")
                
                # 处理每个I帧
                iframe_results = []
                for idx, frame_path in enumerate(i_frames):
                    frame = cv2.imread(str(frame_path))
                    if frame is None:
                        logger.warning(f"跳过损坏的I帧：{frame_path}")
                        continue
                    
                    # 预处理
                    if self.fix_size:
                        frame = fix_image_size(frame)
                    
                    # 模糊检测（基于局部区域）
                    blur_map, global_score, _ = estimate_blur(frame, threshold=self.threshold)
                    is_blurry, blurry_ratio = self._is_blurry_by_area(blur_map)
                    
                    # 记录单I帧结果
                    iframe_result = {
                        'iframe_idx': idx,
                        'global_score': float(global_score),
                        'blurry_area_ratio': float(blurry_ratio),
                        'blurry': is_blurry
                    }
                    
                    # 保存I帧模糊映射图
                    if self.save_blurmap:
                        blurmap_path = output_doc_dir / f"{video_path.stem}_iframe{idx}_blurmap.png"
                        if self._save_blur_map(blur_map, blurmap_path):
                            iframe_result['blurmap_path'] = str(blurmap_path)
                    
                    iframe_results.append(iframe_result)
                
                # 计算模糊I帧比例，判定视频是否模糊
                total_iframe = len(iframe_results)
                if total_iframe == 0:
                    raise ValueError("所有I帧处理失败，无法判定视频模糊性")
                
                blur_iframe_count = sum(1 for res in iframe_results if res['blurry'])
                blur_ratio = blur_iframe_count / total_iframe
                video_blurry = blur_ratio > self.blur_iframe_ratio
                
                # 构造视频总结果
                result = {
                    'type': 'video',
                    'input_path': str(video_path),
                    'output_path': str(output_file_path),
                    'total_i_frames': total_iframe,
                    'blur_i_frame_count': blur_iframe_count,
                    'blur_i_frame_ratio': float(blur_ratio),
                    'blurry': video_blurry,
                    'threshold_used': self.threshold,
                    'area_ratio_used': self.area_ratio,
                    'blur_iframe_ratio_used': self.blur_iframe_ratio,
                    'i_frames': iframe_results,
                    'timestamp': datetime.now().isoformat()
                }
                
                # 若不模糊，移动视频到输出目录（保持目录结构）
                if not video_blurry:
                    output_doc_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(video_path), str(output_file_path))
                    self.stats["non_blurry_kept"] += 1
                    logger.info(f"非模糊视频已保存至：{output_file_path}")
                else:
                    self.stats["blurry_detected"] += 1
                    logger.info(f"检测到模糊视频：{video_path}")
                
                # 保存JSON结果
                if self.save_json:
                    json_path = output_doc_dir / f"{video_path.stem}_blur_result.json"
                    output_doc_dir.mkdir(parents=True, exist_ok=True)  
                    with open(json_path, 'w') as f:
                        json.dump(result, f, indent=4)
                    result['json_path'] = str(json_path)
                
                return result
        
        except Exception as e:
            error_msg = f"处理视频 {video_path} 失败：{str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.stats["processing_errors"] += 1
            return {
                'type': 'video',
                'input_path': str(video_path),
                'status': 'error',
                'error_message': error_msg,
                'timestamp': datetime.now().isoformat()
            }

    def _find_media_files(self):
        """查找所有媒体文件（图片+视频，支持多级目录）"""
        media_files = []  # 元素格式：(文件路径, 类型('image'/'video'))
        for root, _, files in os.walk(self.input_root):
            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()
                if ext in IMAGE_EXTENSIONS:
                    media_files.append((file_path, 'image'))
                elif ext in VIDEO_EXTENSIONS:
                    media_files.append((file_path, 'video'))
        return media_files

    def process(self, params=None, module_config=None):
        """
        执行模糊检测处理
        
        参数:
            params: 包含视频信息等的参数字典
            module_config: 模块配置字典
            
        返回:
            处理结果汇总字典
        """
        try:
            # 查找所有媒体文件
            media_files = self._find_media_files()
            total_files = len(media_files)
            self.stats["total_processed"] = total_files
            logger.info(f"共发现 {total_files} 个媒体文件待处理")
            
            # 处理每个媒体文件
            for media_path, media_type in media_files:
                logger.info(f"开始处理：{media_path}（类型：{media_type}）")
                
                if media_type == 'image':
                    result = self._process_image(media_path)
                    self.stats["image_processed"] += 1
                else:  # 视频
                    result = self._process_video(media_path)
                    self.stats["video_processed"] += 1
                
                self.process_results.append(result)
            
            logger.info("所有媒体文件处理完成!")
            
            # 构建返回结果
            return {
                "status": "completed",
                "input_root_dir": str(self.input_root),
                "output_root_dir": str(self.output_root),
                "stats": self.stats,
                "process_results": self.process_results,
                "parameters": {
                    "threshold": self.threshold,
                    "area_ratio": self.area_ratio,
                    "blur_iframe_ratio": self.blur_iframe_ratio,
                    "fix_size": self.fix_size,
                    "save_json": self.save_json,
                    "save_blurmap": self.save_blurmap
                },
                "environment_info": self.check_environment()
            }
            
        except Exception as e:
            error_msg = f"处理过程中发生错误：{str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return {
                "status": "failed",
                "error_message": error_msg,
                "stats": self.stats,
                "environment_info": self.check_environment()
            }
