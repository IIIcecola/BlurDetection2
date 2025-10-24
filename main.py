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
from blur_detection.blur_detection import BlurDetector

import argparse
from pathlib import Path
from blur_detector import BlurDetector  # 假设你的BlurDetector类定义在blur_detector.py中


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='模糊检测工具（支持图片和视频）',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # 必选参数：输入/输出目录
    parser.add_argument('input_dir', type=str, help='媒体文件根目录（包含图片/视频）')
    parser.add_argument('output_dir', type=str, help='结果输出根目录（保持原目录结构）')
    
    # 分块检测参数
    parser.add_argument('--patch-size', type=int, default=64, 
                      help='分块大小（像素）')
    parser.add_argument('--patch-threshold', type=float, default=30.0, 
                      help='块模糊判断基础阈值（拉普拉斯方差）')
    parser.add_argument('--local-range', type=float, default=0.3, 
                      help='局部阈值浮动比例（±百分比，如0.3表示±30%）')
    parser.add_argument('--min-region-size', type=int, default=9, 
                      help='最小有效模糊区域的块数量')
    
    # 视频检测参数
    parser.add_argument('--blur-iframe-ratio', type=float, default=0.3, 
                      help='视频模糊判定的I帧比例阈值')
    
    # 通用参数
    parser.add_argument('--variable-size', action='store_true', 
                      help='不固定图像尺寸（默认固定为200万像素以保证一致性）')
    parser.add_argument('--no-save-json', action='store_false', dest='save_json', 
                      help='不保存检测结果JSON文件')
    parser.add_argument('--no-save-blurmap', action='store_false', dest='save_blurmap', 
                      help='不保存模糊映射图和标记图')
    
    args = parser.parse_args()

    # 验证输入目录存在
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"错误：输入目录不存在 - {input_path}")
        return

    # 初始化模糊检测器
    detector = BlurDetector(
        input_root_dir=args.input_dir,
        output_root_dir=args.output_dir,
        patch_size=args.patch_size,
        patch_threshold=args.patch_threshold,
        local_threshold_range=args.local_range,
        min_blur_region_size=args.min_region_size,
        blur_iframe_ratio=args.blur_iframe_ratio,
        variable_size=args.variable_size,
        save_json=args.save_json,
        save_blurmap=args.save_blurmap
    )

    # 执行检测
    result = detector.process()

    # 输出处理结果摘要
    print("\n" + "="*50)
    print("处理完成！结果摘要：")
    print(f"状态：{result['status']}")
    print(f"输入目录：{result['input_root_dir']}")
    print(f"输出目录：{result['output_root_dir']}")
    print(f"总处理文件数：{result['stats']['total_processed']}")
    print(f"图片处理数：{result['stats']['image_processed']}")
    print(f"视频处理数：{result['stats']['video_processed']}")
    print(f"检测到模糊文件数：{result['stats']['blurry_detected']}")
    print(f"保留非模糊文件数：{result['stats']['non_blurry_kept']}")
    print(f"处理错误数：{result['stats']['processing_errors']}")
    print("="*50 + "\n")

    if result['status'] == 'failed':
        print(f"错误详情：{result['error_message']}")


if __name__ == "__main__":
    main()
