# 文件头注释，提供版权信息
# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""在线3D深度视频处理管线。

- 连接到RGBD摄像头或RGBD视频文件（当前支持RealSense摄像头和bag文件格式）。
- 捕获/读取颜色和深度帧。允许从摄像头录制。
- 将帧转换为点云，可选计算法线。
- 可视化点云视频和结果。
- 为选定帧保存点云和RGBD图像。

本示例要求Open3D编译时开启 -DBUILD_LIBREALSENSE=ON
"""

# 导入所需的库
import os
import json
import time
import logging as log
import argparse
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import shutil
# 定义摄像头和处理的类
class PipelineModel:
    """控制IO（摄像头，视频文件，录制，保存帧）。方法在工作线程中运行。"""

    def __init__(self, update_view, camera_config_file=None, rgbd_video=None, device=None):
        """初始化。
        Args:
            update_view (callback): 回调函数，用于更新帧的显示元素。
            camera_config_file (str): 摄像头配置的json文件。
            rgbd_video (str): 包含RGBD视频的RS bag文件。如果提供，将忽略连接的摄像头。
            device (str): 计算设备（例如：'cpu:0' 或 'cuda:0'）。
        """
        self.update_view = update_view
        if device:
            self.device = device.lower()
        else:
            self.device = 'cuda:0' if o3d.core.cuda.is_available() else 'cpu:0'
        print(self.device)
        self.o3d_device = o3d.core.Device(self.device)

        self.video = None
        self.camera = None
        self.flag_capture = False
        self.cv_capture = threading.Condition()  # 条件变量
        self.recording = False  # 当前是否正在录制
        self.flag_record = False  # 请求开始/停止录制
        if rgbd_video:  # 视频文件
            self.video = o3d.t.io.RGBDVideoReader.create(rgbd_video)
            self.rgbd_metadata = self.video.metadata
            self.status_message = f"Video {rgbd_video} opened."

        else:  # RGBD摄像头
            now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"{now}.bag"
            self.camera = o3d.t.io.RealSenseSensor()
            if camera_config_file:
                with open(camera_config_file) as ccf:
                    self.camera.init_sensor(o3d.t.io.RealSenseSensorConfig(
                        json.load(ccf)),
                        filename=filename)
            else:
                self.camera.init_sensor(filename=filename)
            self.camera.start_capture(start_record=False)
            self.rgbd_metadata = self.camera.get_metadata()
            intrinsics_matrix = self.rgbd_metadata.intrinsics.intrinsic_matrix
            np.save('config/intrinsics_matrix.npy', intrinsics_matrix)
            np.save('config/width.npy', self.rgbd_metadata.width)
            np.save('config/height.npy', self.rgbd_metadata.height)

            self.status_message = f"Camera {self.rgbd_metadata.serial_number} opened."

        log.info(self.rgbd_metadata)

        # RGBD -> PCD 转换设置
        self.extrinsics = o3d.core.Tensor.eye(4,
                                              dtype=o3d.core.Dtype.Float32,
                                              device=self.o3d_device)
        self.intrinsic_matrix = o3d.core.Tensor(
            self.rgbd_metadata.intrinsics.intrinsic_matrix,
            dtype=o3d.core.Dtype.Float32,
            device=self.o3d_device)
        self.depth_max = 3.0  # m 深度上限
        self.pcd_stride = 2  # 点云降采样，可能提高帧率
        self.flag_normals = False
        self.flag_save_rgbd = False
        self.flag_save_pcd = False

        self.pcd_frame = None
        self.rgbd_frame = None
        self.executor = ThreadPoolExecutor(max_workers=3,
                                           thread_name_prefix='Capture-Save')
        self.flag_exit = False

        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.005,  # 体素长度（米），约 1cm
            sdf_trunc=0.5,  # sdf 截断距离（米），约几个体素长度
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)  # 设置颜色类型为 RGB8

        self.source_down = []
        self.voxel_size = 0.01
        self.transform_0 = np.eye(4)
        base_directory = "./"
        self.base_directory = base_directory
        self.image_depth_dir = os.path.join(base_directory, "images_depths")
        self.pcd_dir = os.path.join(base_directory, "pcds")
        self.image_depth_counter = 1
        self.pcd_counter = 1
        self.ensure_directory_clean()

    @property
    def max_points(self):
        """计算摄像头或RGBD视频分辨率下的单帧最大点数。"""
        return self.rgbd_metadata.width * self.rgbd_metadata.height

    @property
    def vfov(self):
        """获取摄像头或RGBD视频的垂直视场角。"""
        return np.rad2deg(2 * np.arctan(self.intrinsic_matrix[1, 2].item() /
                                        self.intrinsic_matrix[1, 1].item()))
    def ensure_directory_clean(self):
        """确保目标文件夹存在且为空。"""
        for directory in [self.image_depth_dir, self.pcd_dir]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)
    def run(self):
        """运行管线。"""
        n_pts = 0
        frame_id = 0
        t1 = time.perf_counter()
        if self.video:
            self.rgbd_frame = self.video.next_frame()
        else:
            self.rgbd_frame = self.camera.capture_frame(
                wait=True, align_depth_to_color=True)

        pcd_errors = 0
        while (not self.flag_exit and
               (self.video is None or  # 摄像头
                (self.video and not self.video.is_eof()))):  # 视频
            if self.video:
                future_rgbd_frame = self.executor.submit(self.video.next_frame)
            else:
                future_rgbd_frame = self.executor.submit(
                    self.camera.capture_frame,
                    wait=True,
                    align_depth_to_color=True)

            if self.flag_save_pcd:
                self.save_pcd()
                self.flag_save_pcd = False
            try:
                self.rgbd_frame = self.rgbd_frame.to(self.o3d_device)
                self.pcd_frame = o3d.t.geometry.PointCloud.create_from_rgbd_image(
                    self.rgbd_frame, self.intrinsic_matrix, self.extrinsics,
                    self.rgbd_metadata.depth_scale, self.depth_max,
                    self.pcd_stride, self.flag_normals)
                depth_in_color = self.rgbd_frame.depth.colorize_depth(
                    self.rgbd_metadata.depth_scale, 0, self.depth_max)
            except RuntimeError:
                pcd_errors += 1

            if self.pcd_frame.is_empty():
                log.warning(f"No valid depth data in frame {frame_id})")
                continue

            n_pts += self.pcd_frame.point.positions.shape[0]
            if frame_id % 60 == 0 and frame_id > 0:
                t0, t1 = t1, time.perf_counter()
                log.debug(f"\nframe_id = {frame_id}, \t {(t1 - t0) * 1000. / 60:0.2f}"
                          f"ms/frame \t {(t1 - t0) * 1e9 / n_pts} ms/Mp\t")
                n_pts = 0

            self.rgbd_frame = future_rgbd_frame.result()
            self.save_rgbd(self.rgbd_frame)
            self.save_pcd(self.pcd_frame)

            color_image_ = o3d.t.geometry.Image.to_legacy(self.rgbd_frame.color)
            cv2.imshow('color',np.array(color_image_))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_id += 1
        self.executor.shutdown()
        log.debug(f"create_from_depth_image() errors = {pcd_errors}")

    def toggle_record(self):
        if self.camera is not None:
            if self.flag_record and not self.recording:
                self.camera.resume_record()
                self.recording = True
            elif not self.flag_record and self.recording:
                self.camera.pause_record()
                self.recording = False

    def save_pcd(self,pcd_frame):
        """单独保存点云数据。"""
        base_filename = str(self.pcd_counter)
        pcd_filename = os.path.join(self.pcd_dir, f"{base_filename}.ply")

        # 转换颜色为 uint8 以确保兼容性
        pcd_frame.point['colors'] = (pcd_frame.point['colors'] * 255).to(o3d.core.Dtype.UInt8)

        # 保存点云数据
        o3d.t.io.write_point_cloud(pcd_filename, pcd_frame, write_ascii=False, compressed=True)

        # 更新文件编号
        self.pcd_counter += 1
        print(f"Saved point cloud to {pcd_filename}")

    def save_rgbd(self,rgbd_frame):
        """单独保存RGB和深度图像。"""
        base_filename = str(self.image_depth_counter)
        image_filename = os.path.join(self.image_depth_dir, f"{base_filename}.jpg")
        depth_filename = os.path.join(self.image_depth_dir, f"{base_filename}.png")

        # 保存图像和深度图
        o3d.t.io.write_image(image_filename, rgbd_frame.color)
        o3d.t.io.write_image(depth_filename, rgbd_frame.depth)

        # 更新文件编号
        self.image_depth_counter += 1
        print(f"Saved RGB image to {image_filename} and depth image to {depth_filename}")


# 应用的入口点类。控制 PipelineModel 对象进行 IO 和处理，以及 PipelineView 对象进行显示和 UI。所有方法都在主线程上操作。
class PipelineController:
    """应用的入口点。控制 PipelineModel 对象进行 IO 和处理，以及 PipelineView 对象进行显示和 UI。所有方法都在主线程上操作。
    """

    def __init__(self, camera_config_file=None, rgbd_video=None, device=None):
        """初始化。
        Args:
            camera_config_file (str): 摄像头配置 json 文件。
            rgbd_video (str): 包含 RGBD 视频的 RS bag 文件。如果提供，将忽略连接的摄像头。
            device (str): 计算设备（例如：'cpu:0' 或 'cuda:0'）。
        """
        self.pipeline_model = PipelineModel(self.update_view,
                                            camera_config_file, rgbd_video,
                                            device)
        self.pipeline_model.run()

    def update_view(self, frame_elements):
        """使用新数据更新视图。可以从任何线程调用。
        Args:
            frame_elements (dict): 显示元素（点云和图像）
                从新帧显示的新数据。
        """
        gui.Application.instance.post_to_main_thread(
            self.pipeline_view.window,
            lambda: self.pipeline_view.update(frame_elements))

    def on_toggle_capture(self, is_enabled):
        """切换捕捉的回调函数。"""
        self.pipeline_model.flag_capture = is_enabled
        if not is_enabled:
            self.on_toggle_record(False)
            if self.pipeline_view.toggle_record is not None:
                self.pipeline_view.toggle_record.is_on = False
        else:
            with self.pipeline_model.cv_capture:
                self.pipeline_model.cv_capture.notify()

    def on_toggle_record(self, is_enabled):
        """切换录制 RGBD 视频的回调函数。"""
        self.pipeline_model.flag_record = is_enabled

    def on_toggle_normals(self, is_enabled):
        """切换显示法线的回调函数"""
        self.pipeline_model.flag_normals = is_enabled
        self.pipeline_view.flag_normals = is_enabled
        self.pipeline_view.flag_gui_init = False

    def on_window_close(self):
        """用户关闭应用程序窗口时的回调函数。"""
        self.pipeline_model.flag_exit = True
        with self.pipeline_model.cv_capture:
            self.pipeline_model.cv_capture.notify_all()
        return True  # 确认可以关闭窗口

    def on_save_pcd(self):
        """保存当前点云的回调函数。"""
        self.pipeline_model.flag_save_pcd = True

    def on_save_rgbd(self):
        """保存当前 RGBD 图像对的回调函数。"""
        self.pipeline_model.flag_save_rgbd = True


if __name__ == "__main__":

    log.basicConfig(level=log.INFO)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--camera-config',default='./config/camera.json',
                        help='RGBD摄像头配置JSON文件')
    parser.add_argument('--rgbd-video', help='RGBD视频文件（RealSense bag）')
    parser.add_argument('--device',
                        help='运行计算的设备。例如 cpu:0 或 cuda:0 '
                             '如果可用，默认使用CUDA GPU，否则使用CPU。')

    args = parser.parse_args()
    if args.camera_config and args.rgbd_video:
        log.critical(
            "请仅提供 --camera-config 和 --rgbd-video 参数中的一个"
        )
    else:
        PipelineController(args.camera_config, args.rgbd_video, args.device)
