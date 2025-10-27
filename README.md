该仓库是一个基于OpenCV和NumPy的**图像与视频模糊检测工具**，主要用于评估媒体文件的模糊程度，适用于照片筛选、视频质量检测等场景。以下是其核心信息：


### 1. 核心功能
- **图像模糊检测**：通过计算图像拉普拉斯算子的总方差（total variance of the laplacian）评分，分数越高表示图像越清晰，低于阈值则判定为模糊。
- **批量处理能力**：支持单张图片、多级目录下的多张图片批量检测。
- **视频模糊检测**：通过`ffmpeg`提取视频中的I帧（关键帧），分析I帧的模糊程度，结合模糊I帧占比判断视频整体是否模糊。
- **结果输出**：可将检测结果（模糊评分、是否模糊等）保存为JSON文件，也能生成并保存“模糊映射图”（直观展示图像中模糊区域）。
- **灵活参数配置**：支持自定义模糊阈值、图像尺寸是否固定、模糊区域占比阈值等参数。


### 2. 实现原理
基于拉普拉斯算子的边缘检测特性：清晰图像的边缘信息丰富，拉普拉斯算子的方差较大；模糊图像的边缘信息少，方差较小。通过设定阈值，可判断图像是否模糊。对于视频，通过分析关键帧（I帧）的模糊比例来评估整体质量。


### 3. 主要文件与模块
- **核心检测逻辑**：`blur_detection/detection.py`，包含图像尺寸调整、模糊评分计算、模糊映射图生成等基础函数。
- **增强检测逻辑**：`blur_detection/blur_detection.py`，实现分块检测、局部模糊区域分析（通过连通区域过滤小面积模糊块）等进阶功能。
- **基础脚本**：`process.py`，支持图像批量检测、结果保存、模糊映射图显示等基础操作。
- **增强脚本**：`main.py`，扩展支持视频处理（提取I帧）、基于局部模糊区域占比的精准判断等功能。
- **辅助文件**：
  - `requirements.txt`：依赖库（`numpy`、`opencv-python`）。
  - `Makefile`：提供依赖安装（`make install_deps`）、测试（`make test`）、代码格式化（`make yapf`）等快捷命令。
  - `.travis.yml`：配置持续集成，支持多版本Python（2.7、3.4+）测试。
  - `LICENSE`：采用MIT许可证，允许自由使用、修改和分发。


### 4. 使用方法
- **安装依赖**：
  ```bash
  # 直接安装
  pip install -U -r requirements.txt
  # 或通过Makefile
  make install_deps
  ```

- **基础图像检测（`process.py`）**：
  ```bash
  # 检测单张图片
  python process.py -i input_image.png
  # 检测目录下所有图片（含子目录）
  python process.py -i input_directory/
  # 保存结果到JSON文件
  python process.py -i input_directory/ -s results.json
  # 显示模糊映射图
  python process.py -i input_directory/ -d
  ```

- **增强功能（`main.py`，支持视频和局部模糊判断）**：
  ```bash
  # 处理视频（提取I帧分析）
  python main.py input_video.mp4 output_results/
  # 自定义模糊区域占比阈值（如>20%判定为模糊）
  python main.py input_images/ output_results/ -a 0.2
  ```


该工具基于Adrian Rosebrock的《Blur Detection With Opencv》实现，适合需要批量评估图像/视频清晰度的场景。
