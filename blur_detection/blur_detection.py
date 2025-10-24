import sys
import os
import logging
import json
import shutil
import traceback
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from collections import deque

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
                 patch_size=64,
                 patch_threshold=30.0,
                 local_threshold_range=0.3,
                 min_blur_region_size=9,
                 blur_iframe_ratio=0.3, 
                 variable_size=False,
                 save_json=True, 
                 save_blurmap=True):
        """
        初始化模糊检测工具（增强版：支持分块检测和区域过滤）
        
        新增参数:
        参数:
            input_root_dir: 输入媒体文件根目录
            output_root_dir: 输出结果根目录
            patch_size: 分块大小（像素）
            patch_threshold: 块模糊判断基础阈值（拉普拉斯方差）
            local_threshold_range: 局部阈值浮动比例（±%）
            min_blur_region_size: 最小有效模糊区域的块数量
            blur_iframe_ratio: 视频模糊I帧比例阈值（默认：0.3）
            variable_size: 是否不固定图像尺寸（默认：False，即固定为200万像素）
            save_json: 是否保存检测结果JSON（默认：True）
            save_blurmap: 是否保存模糊映射图（默认：True）
        """
        self.input_root = Path(input_root_dir).resolve()
        self.output_root = Path(output_root_dir).resolve()
        
        # 分块检测参数
        self.patch_size = patch_size
        self.patch_threshold = patch_threshold
        self.local_range = local_threshold_range
        self.min_region_size = min_blur_region_size
        
        # 原有参数
        self.blur_iframe_ratio = blur_iframe_ratio
        self.fix_size = not variable_size
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
        logger.info(f"分块参数: 大小={patch_size}px, 基础阈值={patch_threshold}, "
                    f"局部浮动={local_threshold_range*100}%, 最小区域块数={min_blur_region_size}")
        logger.info(f"视频模糊I帧比例阈值: {blur_iframe_ratio}")
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
            relative_path = input_path.relative_to(self.input_root)
            output_file_path = self.output_root / relative_path
            output_doc_dir = output_file_path.parent
            return output_file_path, output_doc_dir
        except ValueError as e:
            logger.error(f"计算输出路径失败: {str(e)}")
            raise

    def _save_blur_map(self, blur_map, save_path):
        """保存模糊映射图"""
        try:
            pretty_map = pretty_blur_map(blur_map)
            pretty_map = ((pretty_map - pretty_map.min()) / 
                         (pretty_map.max() - pretty_map.min()) * 255).astype(np.uint8)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), pretty_map)
            logger.debug(f"模糊映射图已保存至：{save_path}")
            return True
        except Exception as e:
            logger.error(f"保存模糊映射图失败：{str(e)}\n{traceback.format_exc()}")
            return False

    # -------------------------- 新增分块检测核心方法 --------------------------
    def _divide_into_patches(self, image):
        """划分图像为块（保留边缘不完整块）"""
        h, w = image.shape[:2]
        patches = []  # 元素: (块坐标(row, col), 块数据, 边界框(x1,y1,x2,y2))
        
        for row in range(0, h, self.patch_size):
            for col in range(0, w, self.patch_size):
                y1 = row
                y2 = min(row + self.patch_size, h)
                x1 = col
                x2 = min(col + self.patch_size, w)
                patch = image[y1:y2, x1:x2]
                patches.append(((row, col), patch, (x1, y1, x2, y2)))
        
        return patches

    def _calculate_local_patch_threshold(self, patch):
        """基于块亮度计算局部阈值（亮区域提高阈值，暗区域降低阈值）"""
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
        brightness = np.mean(gray)
        
        if brightness > 200:  # 高亮区域
            return self.patch_threshold * (1 + self.local_range)
        elif brightness < 50:  # 暗区域
            return self.patch_threshold * (1 - self.local_range)
        return self.patch_threshold  # 中间亮度

    def _find_connected_regions(self, blur_patches):
        """找到相邻的模糊块（8邻域连通）"""
        if not blur_patches:
            return []
        visited = set()
        regions = []
        
        for patch in blur_patches:
            if patch not in visited:
                # BFS寻找连通区域
                queue = deque([patch])
                visited.add((patch))
                region = {patch}
                
                while queue:
                    r, c = queue.pop(0)
                    # 检查8个方向的邻居
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            neighbor = (r + dr, c + dc)
                            if neighbor in blur_patches and neighbor not in visited:
                                visited.add(neighbor)
                                region.add(neighbor)
                                queue.append(neighbor)
                
                regions.append(region)
        
        return regions

    def _is_blurry_by_patches(self, image):
        """基于分块检测判断是否模糊（替代原像素级判断）"""
        try:
            # 1. 划分块
            patches = self._divide_into_patches(image)
            total_patches = len(patches)
            if total_patches == 0:
                return True, 1.0, [], []  # 空图像视为模糊
            
            # 2. 检测每个块是否模糊
            blur_patches = set()
            patch_details = []
            for (coord, patch, bbox) in patches:
                # 计算块的拉普拉斯方差
                gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
                laplacian = cv2.Laplacian(gray_patch, cv2.CV_64F)
                patch_var = np.var(laplacian)
                
                # 局部阈值判断
                local_thresh = self._calculate_local_patch_threshold(patch)
                is_blur = bool(patch_var < local_thresh)
                
                if is_blur:
                    blur_patches.add(coord)
                
                patch_details.append({
                    "coord": coord,
                    "bbox": bbox,
                    "lapiacian_variance": float(patch_var),
                    "local_threshold": float(local_thresh),
                    "is_blur": is_blur
                })
            logger.debug(f"总模糊块数量: {len(blur_patches)}")
            # 3. 连通区域分析并过滤小区域
            regions = self._find_connected_regions(blur_patches)
            for i, region in enumerate(regions):
                logger.debug(f"连通区域{i+1}包含块数量: {len(region)}")
            valid_regions = [r for r in regions if len(r) >= self.min_region_size]
            valid_blur_patches = set()
            for region in valid_regions:
                valid_blur_patches.update(region)
            
            # 4. 计算有效模糊块占比
            valid_blur_count = len(valid_blur_patches)
            blurry_ratio = valid_blur_count / total_patches if total_patches > 0 else 1.0
            
            # 5. 整理区域信息（用于结果展示）
            region_details = []
            for region in valid_regions:
                bboxes = [p["bbox"] for p in patch_details if p["coord"] in region]
                x1 = min(b[0] for b in bboxes)
                y1 = min(b[1] for b in bboxes)
                x2 = max(b[2] for b in bboxes)
                y2 = max(b[3] for b in bboxes)
                region_details.append({
                    "bbox": (x1, y1, x2, y2),
                    "patch_count": len(region),
                    "patches": list(region)
                })
            
            return (blurry_ratio > self.min_region_size / total_patches), blurry_ratio, region_details, patch_details
        
        except Exception as e:
            logger.error(f"分块模糊判断失败：{str(e)}\n{traceback.format_exc()}")
            return True, 1.0, [], []
    # ------------------------------------------------------------------------

    def _process_image(self, image_path):
        """处理单张图片（使用分块检测）"""
        try:
            # 读取图片
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("无法读取图片文件")
            
            # 预处理
            original_image = image.copy()
            if self.fix_size:
                image = fix_image_size(image)
            
            # 模糊检测（分块版本）
            blur_map, global_score, _ = estimate_blur(image)  # 保留全局评分用于参考
            is_blurry, blurry_ratio, region_details, patch_details = self._is_blurry_by_patches(image)
            
            # 计算输出路径
            output_file_path, output_doc_dir = self._get_output_paths(image_path)
            
            # 构造结果字典
            result = {
                'type': 'image',
                'input_path': str(image_path),
                'output_path': str(output_file_path),
                'global_score': float(global_score),  # 保留全局评分
                'blurry_patch_ratio': float(blurry_ratio),  # 替换原像素占比
                'blurry': is_blurry,
                'patch_parameters': {
                    'patch_size': self.patch_size,
                    'base_threshold': self.patch_threshold,
                    'local_range': self.local_range,
                    'min_region_size': self.min_region_size
                },
                'blur_regions': region_details,  # 新增模糊区域详情
                'patch_details': patch_details,
                'timestamp': datetime.now().isoformat()
            }
            
            # 保存带模糊区域标记的图像（增强可视化）
            if self.save_blurmap:
                # 1. 保存原始模糊映射图
                blurmap_path = output_doc_dir / f"{image_path.stem}_blurmap.png"
                if self._save_blur_map(blur_map, blurmap_path):
                    result['blurmap_path'] = str(blurmap_path)
                
                # 2. 保存带区域标记的图像
                marked_image = original_image.copy()
                for region in region_details:
                    x1, y1, x2, y2 = region["bbox"]
                    cv2.rectangle(marked_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                marked_path = output_doc_dir / f"{image_path.stem}_marked.png"
                cv2.imwrite(str(marked_path), marked_image)
                result['marked_path'] = str(marked_path)
            
            # 移动非模糊文件
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
                output_doc_dir.mkdir(parents=True, exist_ok=True)
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
        """提取视频I帧"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", str(video_path),
                "-vf", "select=eq(pict_type\\,I)",
                "-vsync", "vfr",
                f"{output_dir}/i_frame_%04d.jpg"
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
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
        """处理单个视频（I帧采用分块检测）"""
        try:
            output_file_path, output_doc_dir = self._get_output_paths(video_path)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                i_frames = self._extract_i_frames(video_path, tmpdir)
                logger.info(f"从视频 {video_path} 中提取到 {len(i_frames)} 个I帧")
                
                iframe_results = []
                for idx, frame_path in enumerate(i_frames):
                    frame = cv2.imread(str(frame_path))
                    if frame is None:
                        logger.warning(f"跳过损坏的I帧：{frame_path}")
                        continue
                    
                    original_frame = frame.copy()
                    if self.fix_size:
                        frame = fix_image_size(frame)
                    
                    # I帧采用分块检测
                    blur_map, global_score, _ = estimate_blur(frame)
                    is_blurry, blurry_ratio, region_details, patch_details = self._is_blurry_by_patches(frame)
                    
                    iframe_result = {
                        'iframe_idx': idx,
                        'global_score': float(global_score),
                        'blurry_patch_ratio': float(blurry_ratio),
                        'blurry': is_blurry,
                        'blur_regions': region_details,
                        'patch_details': patch_details
                    }
                    
                    # 保存I帧相关可视化结果
                    if self.save_blurmap:
                        # 模糊映射图
                        blurmap_path = output_doc_dir / f"{video_path.stem}_iframe{idx}_blurmap.png"
                        if self._save_blur_map(blur_map, blurmap_path):
                            iframe_result['blurmap_path'] = str(blurmap_path)
                        
                        # 带区域标记的I帧
                        marked_path = output_doc_dir / f"{video_path.stem}_iframe{idx}_marked.png"
                        marked_frame = original_frame.copy()
                        for region in region_details:
                            x1, y1, x2, y2 = region["bbox"]
                            cv2.rectangle(marked_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.imwrite(str(marked_path), marked_frame)
                        iframe_result['marked_path'] = str(marked_path)
                    
                    iframe_results.append(iframe_result)
                
                # 视频整体模糊判断
                total_iframe = len(iframe_results)
                if total_iframe == 0:
                    raise ValueError("所有I帧处理失败，无法判定视频模糊性")
                
                blur_iframe_count = sum(1 for res in iframe_results if res['blurry'])
                blur_ratio = blur_iframe_count / total_iframe
                video_blurry = blur_ratio > self.blur_iframe_ratio
                
                # 构造视频结果
                result = {
                    'type': 'video',
                    'input_path': str(video_path),
                    'output_path': str(output_file_path),
                    'total_i_frames': total_iframe,
                    'blur_i_frame_count': blur_iframe_count,
                    'blur_i_frame_ratio': float(blur_ratio),
                    'blurry': video_blurry,
                    'patch_parameters': {
                        'patch_size': self.patch_size,
                        'base_threshold': self.patch_threshold,
                        'local_range': self.local_range,
                        'min_region_size': self.min_region_size
                    },
                    'i_frames': iframe_results,
                    'timestamp': datetime.now().isoformat()
                }
                
                # 移动非模糊视频
                if not video_blurry:
                    output_doc_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(video_path), str(output_file_path))
                    self.stats["non_blurry_kept"] += 1
                    logger.info(f"非模糊视频已保存至：{output_file_path}")
                else:
                    self.stats["blurry_detected"] += 1
                    logger.info(f"检测到模糊视频：{video_path}")
                
                # 保存JSON
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
        """查找所有媒体文件"""
        media_files = []
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
        """执行模糊检测处理"""
        try:
            media_files = self._find_media_files()
            total_files = len(media_files)
            self.stats["total_processed"] = total_files
            logger.info(f"共发现 {total_files} 个媒体文件待处理")
            
            for media_path, media_type in media_files:
                logger.info(f"开始处理：{media_path}（类型：{media_type}）")
                
                if media_type == 'image':
                    result = self._process_image(media_path)
                    self.stats["image_processed"] += 1
                else:
                    result = self._process_video(media_path)
                    self.stats["video_processed"] += 1
                
                self.process_results.append(result)
            
            logger.info("所有媒体文件处理完成!")
            
            return {
                "status": "completed",
                "input_root_dir": str(self.input_root),
                "output_root_dir": str(self.output_root),
                "stats": self.stats,
                "process_results": self.process_results,
                "parameters": {
                    "patch_size": self.patch_size,
                    "patch_threshold": self.patch_threshold,
                    "local_threshold_range": self.local_range,
                    "min_blur_region_size": self.min_region_size,
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
