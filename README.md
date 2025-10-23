该仓库是一个基于OpenCV和NumPy的**图像模糊检测工具**，主要用于评估图像的模糊程度，同时支持视频的模糊检测（通过提取I帧分析）。以下是其核心信息介绍：


### 1. 核心功能
- **图像模糊检测**：通过计算图像拉普拉斯算子的总方差（total variance of the laplacian）来评分模糊程度，分数越高表示图像越清晰。
- **批量处理**：支持单张图片、目录下的多张图片（含多级目录）的批量检测。
- **视频模糊检测**：通过`ffmpeg`提取视频中的I帧（关键帧），分析I帧的模糊程度，判断视频整体是否模糊。
- **结果输出**：可将检测结果（模糊评分、是否模糊等）保存为JSON文件，也可生成并保存“模糊映射图”（直观展示图像中模糊区域）。
- **参数配置**：支持自定义模糊阈值、图像尺寸是否固定、模糊区域占比阈值等参数。


### 2. 实现原理
基于拉普拉斯算子的边缘检测特性：清晰图像的边缘信息丰富，拉普拉斯算子的方差较大；模糊图像的边缘信息少，方差较小。通过设定阈值，可判断图像是否模糊。


### 3. 主要文件与模块
- **核心检测逻辑**：`blur_detection/detection.py`，包含图像尺寸调整、模糊评分计算、模糊映射图生成等函数。
- **基础脚本**：`process.py`，支持图像的批量检测、结果保存、模糊映射图显示等基础功能。
- **增强脚本**：`main.py`，扩展支持视频处理（提取I帧）、基于局部模糊区域占比的判断（更精准）等功能。
- **辅助文件**：
  - `requirements.txt`：依赖库（`numpy`、`opencv-python`）。
  - `Makefile`：提供依赖安装、测试、代码格式化等快捷命令。
  - `.travis.yml`：配置持续集成，支持多版本Python测试。


### 4. 使用方法
- **安装依赖**：
  ```bash
  pip install -U -r requirements.txt
  # 或通过Makefile：make install_deps
  ```

- **基础图像检测**（`process.py`）：
  ```bash
  # 单张图片
  python process.py -i input_image.png
  # 目录下的图片
  python process.py -i input_directory/
  # 保存结果到JSON
  python process.py -i input_directory/ -s results.json
  # 显示模糊映射图
  python process.py -i input_directory/ -d
  ```

- **增强功能**（`main.py`，支持视频和局部模糊判断）：
  ```bash
  # 处理视频（提取I帧分析）
  python main.py -i video.mp4 -o output_dir
  # 自定义模糊区域占比阈值
  python main.py -i images/ -a 0.2  # 模糊区域占比>20%判定为模糊
  ```


该工具基于Adrian Rosebrock的博客文章《Blur Detection With Opencv》实现，适合需要批量评估图像/视频清晰度的场景（如照片筛选、视频质量检测等）。
