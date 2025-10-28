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
`blur_detection/blur_detection.py`中的`BlurDetector`类是该仓库增强版模糊检测逻辑的核心实现，支持分块检测、局部模糊区域分析、视频I帧处理等进阶功能，相比基础检测逻辑（`detection.py`）更精准。以下是该类的详细说明：


### **5.增强功能类概述**
`BlurDetector`类通过分块检测图像局部模糊区域、结合连通区域分析过滤噪声，并支持视频关键帧（I帧）的批量处理，最终根据模糊区域占比判断媒体文件是否模糊。适用于需要精准识别局部模糊（而非整体模糊）的场景。


#### **初始化方法（`__init__`）**
该方法用于配置检测参数并初始化类属性，核心参数及作用如下：

| 参数名                 | 类型    | 默认值   | 说明                                                                 |
|------------------------|---------|----------|----------------------------------------------------------------------|
| `input_root_dir`       | str     | 必传     | 输入媒体文件（图像/视频）的根目录                                     |
| `output_root_dir`      | str     | 必传     | 输出结果（JSON、模糊映射图）的根目录，保持与输入目录一致的结构         |
| `patch_size`           | int     | 64       | 图像分块大小（像素），将图像划分为多个此尺寸的块进行局部检测           |
| `patch_threshold`      | float   | 30.0     | 块模糊判断的基础阈值（拉普拉斯方差），低于此值的块视为模糊             |
| `local_threshold_range`| float   | 0.3      | 局部阈值浮动比例（±%），根据块亮度动态调整阈值（亮区提高、暗区降低）   |
| `min_blur_region_size` | int     | 9        | 最小有效模糊区域的块数量，过滤小面积（噪声）模糊区域                   |
| `blur_iframe_ratio`    | float   | 0.3      | 视频模糊判定阈值：模糊I帧占比超过此值则判定视频整体模糊               |
| `variable_size`        | bool    | False    | 是否固定图像尺寸（默认`False`：固定为200万像素，保证评分一致性）      |
| `save_json`            | bool    | True     | 是否保存检测结果为JSON文件                                           |
| `save_blurmap`         | bool    | True     | 是否保存模糊映射图（直观展示模糊区域）                               |





#### 分块检测核心方法（增强功能的核心）

##### （1）图像分块（`_divide_into_patches`）
- 功能：将图像按`patch_size`划分为多个块，记录块的索引和像素边界。
- 参数：`image`（OpenCV读取的图像数组）。
- 返回值：列表，每个元素为`( (行索引, 列索引), 块数据, 像素边界框 )`，其中：
  - 行/列索引：块在图像中的网格位置（从0开始）；
  - 块数据：对应区域的图像子数组；
  - 像素边界框：`(x1, y1, x2, y2)`，表示块在原始图像中的像素坐标范围。


##### （2）局部阈值计算（`_calculate_local_patch_threshold`）
- 功能：根据块的亮度动态调整模糊判断阈值（解决亮区/暗区误判问题）。
- 逻辑：
  - 若块亮度（灰度均值）>200（高亮区域）：阈值提高为`patch_threshold * (1 + local_range)`；
  - 若块亮度<50（暗区域）：阈值降低为`patch_threshold * (1 - local_range)`；
  - 中间亮度：使用原始`patch_threshold`。


##### （3）连通区域分析（`_find_connected_regions`）
- 功能：寻找相邻的模糊块（8邻域连通，即上下左右及对角线相邻），过滤小面积噪声。
- 流程：
  1. 输入模糊块的坐标集合（`blur_patches`）；
  2. 用BFS（广度优先搜索）遍历相邻块，形成连通区域；
  3. 返回所有连通区域的集合。


##### （4）分块模糊判断（`_is_blurry_by_patches`）
- 功能：基于分块检测结果判断图像是否模糊（核心逻辑）。
- 流程：
  1. 调用`_divide_into_patches`划分图像为块；
  2. 对每个块计算拉普拉斯方差，结合`_calculate_local_patch_threshold`判断是否为模糊块；
  3. 调用`_find_connected_regions`找到连通的模糊区域，过滤掉块数量小于`min_blur_region_size`的区域；
  4. 计算有效模糊块占总块数的比例（`blurry_ratio`），用于最终判断。
- 返回值：`(是否模糊, 模糊比例, 块详情, 有效模糊区域详情)`。


#### **增强点总结**
相比基础检测逻辑（`process.py`中基于整体拉普拉斯方差的判断），`BlurDetector`类的核心优势在于：
1. **局部检测**：通过分块分析识别图像局部模糊（而非仅判断整体）；
2. **自适应阈值**：结合块亮度动态调整阈值，减少亮区/暗区误判；
3. **噪声过滤**：通过连通区域分析排除小面积噪声模糊；
4. **视频支持**：可处理视频文件（提取I帧后批量检测）。



该工具基于Adrian Rosebrock的《Blur Detection With Opencv》实现，适合需要批量评估图像/视频清晰度的场景。
