# README.md

## 概述

本仓库包含两个与3D深度视频处理相关的Python脚本，它们使用Open3D库进行RGBD视频的处理和点云的生成。

1. `tensor_pcd_load_rgbd.py`：从RGBD摄像头或视频文件加载数据并生成点云。
2. `tensor_pcd_record_rbgd.py`：从RGBD摄像头或视频文件捕获数据，并将其记录为RGBD图像和点云。

## 依赖项

该项目依赖以下Python包：
- `open3d`：用于3D数据处理和可视化。
- `numpy`：用于数值运算。
- `opencv-python`：用于图像处理。
- `argparse`：用于解析命令行参数。
- `concurrent.futures`：用于并发执行。
- `logging`：用于日志记录。

## 安装

使用pip安装必要的包：

```bash
pip install open3d numpy opencv-python argparse
```

## 使用方法

### tensor_pcd_load_rgbd.py

`tensor_pcd_load_rgbd.py`脚本用于从RGBD摄像头或视频文件加载数据，并生成点云。

#### 运行示例

1. 确保配置文件和视频文件已准备好。
2. 运行脚本：

```bash
python tensor_pcd_load_rgbd.py --camera-config ./config/camera.json --rgbd-video ./path/to/video.bag --device cuda:0
```

#### 脚本功能
- 连接到RGBD摄像头或视频文件。
- 捕获颜色和深度帧。
- 将帧转换为点云。
- 可视化点云并保存生成的点云和RGBD图像。

### tensor_pcd_record_rbgd.py

`tensor_pcd_record_rbgd.py`脚本用于从RGBD摄像头或视频文件捕获数据，并将其记录为RGBD图像和点云。

#### 运行示例

1. 确保配置文件和视频文件已准备好。
2. 运行脚本：

```bash
python tensor_pcd_record_rbgd.py --camera-config ./config/camera.json --rgbd-video ./path/to/video.bag --device cuda:0
```

#### 脚本功能
- 连接到RGBD摄像头或视频文件。
- 捕获和记录颜色和深度帧。
- 将帧转换为点云。
- 保存点云和RGBD图像。

## 代码结构

### tensor_pcd_load_rgbd.py

- `PipelineModel` 类：处理数据的加载、点云的生成和可视化。
- `PipelineController` 类：控制管道模型的运行和管理用户界面。

### tensor_pcd_record_rbgd.py

- `PipelineModel` 类：处理数据的捕获、点云的生成和保存。
- `PipelineController` 类：控制管道模型的运行和管理用户界面。

## 贡献

欢迎提出问题或提交改进或错误修复的拉取请求。
