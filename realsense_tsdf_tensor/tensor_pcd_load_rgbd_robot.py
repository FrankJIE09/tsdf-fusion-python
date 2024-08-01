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
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


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
        if device:
            self.device = device.lower()
        else:
            self.device = 'cuda:0' if o3d.core.cuda.is_available() else 'cpu:0'
        print(self.device)
        self.o3d_device = o3d.core.Device(self.device)
        self.intrinsics_matrix = np.load('config/intrinsics_matrix.npy')
        self.width = np.load('config/width.npy')
        self.height = np.load('config/height.npy')

        self.camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()

        self.camera_intrinsics.set_intrinsics(width=self.width, height=self.height,
                                              fx=self.intrinsics_matrix[0, 0], fy=self.intrinsics_matrix[1, 1],
                                              cx=self.intrinsics_matrix[0, 2],
                                              cy=self.intrinsics_matrix[1, 2])  # RGBD -> PCD 转换设置
        self.camera_extrinsics = np.linalg.inv(np.array([
            [0.01527133, -0.71587985, -0.69805646, -0.04416701],
            [0.99984605, 0.01696656, 0.0044738, -0.02317145],
            [0.00864091, -0.69801732, 0.71602874, 0.01392886],
            [0., 0., 0., 1.]
        ], dtype=np.float64))

        self.intrinsic_matrix = o3d.core.Tensor(
            self.intrinsics_matrix,
            dtype=o3d.core.Dtype.Float32,
            device=self.o3d_device)

        self.pcd_frame = None
        self.rgbd_frame = None
        self.executor = ThreadPoolExecutor(max_workers=3,
                                           thread_name_prefix='Capture-Save')
        self.flag_exit = False
        self.voxel_size = 0.02

        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.002,  # 体素长度（米），约 1cm
            sdf_trunc=0.1,  # sdf 截断距离（米），约几个体素长度
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)  # 设置颜色类型为 RGB8

        self.source_down = []
        self.transform_0 = np.eye(4)

        self.base_directory = "./"
        self.image_depth_dir = os.path.join(self.base_directory, "images_depths")
        self.robot_pose_dir = os.path.join(self.base_directory, "T_data")

        self.pcd_dir = os.path.join(self.base_directory, "pcds")
        self.init_robot_transform = np.eye(4)
        self.last_robot_transform = np.eye(4)

    @property
    def max_points(self):
        """计算摄像头或RGBD视频分辨率下的单帧最大点数。"""
        return self.width * self.height

    @property
    def vfov(self):
        """获取摄像头或RGBD视频的垂直视场角。"""
        return np.rad2deg(2 * np.arctan(self.intrinsic_matrix[1, 2].item() /
                                        self.intrinsic_matrix[1, 1].item()))

    def run(self):
        frame_id = 0
        # 列出目录中的所有文件
        files = os.listdir(self.image_depth_dir)

        # 计算扩展名为.png的文件数量
        png_count = sum(
            1 for file in files if
            os.path.isfile(os.path.join(self.image_depth_dir, file)) and file.lower().endswith('.png'))
        transforms_array = []  # 创建一个空列表来存储每次循环的数组
        transforms_array.append(np.eye(4))
        while frame_id < png_count:
            if frame_id % 10 != 0:
                frame_id += 1
                continue
            pcd_path = os.path.join(self.pcd_dir, f"{frame_id + 1}.ply")
            self.pcd_frame = o3d.t.io.read_point_cloud(pcd_path, )
            self.pcd_frame = self.pcd_frame.cuda()
            if frame_id == 0:
                robot_transform_path = os.path.join(self.robot_pose_dir, f"T_base2ee0{frame_id}.npz")
                self.init_robot_transform = np.load(robot_transform_path)['arr_0'] @ self.camera_extrinsics
                robot_transform = np.load(robot_transform_path)['arr_0'] @ self.camera_extrinsics
                frame_id += 1
                self.source_down = self.pcd_frame
                self.last_robot_transform = robot_transform
                continue
            robot_transform_path = os.path.join(self.robot_pose_dir, f"T_base2ee0{frame_id}.npz")
            robot_transform = np.load(robot_transform_path)['arr_0'] @ self.camera_extrinsics
            source_to_target_transform = np.linalg.inv(self.last_robot_transform) @ robot_transform
            target_down = self.pcd_frame
            result_ransac = o3d.t.pipelines.registration.icp(self.source_down, target_down,
                                                             init_source_to_target=source_to_target_transform,
                                                             estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
                                                             max_correspondence_distance=self.voxel_size * 1.5,
                                                             voxel_size=self.voxel_size,
                                                             criteria=o3d.t.pipelines.registration.ICPConvergenceCriteria(
                                                                 max_iteration=10000))
            self.source_down = self.pcd_frame
            self.last_robot_transform = robot_transform

            transform = result_ransac.transformation  # 获取变换矩阵

            transformation_float_values = np.eye(4)
            # 遍历张量的每个元素，并将其转换为 float
            for m in range(transform.shape[0]):
                for n in range(transform.shape[1]):
                    transformation_float_values[m][n] = transform[m][n].item()

            image_path = os.path.join(self.image_depth_dir, f"{frame_id + 1}.jpg")
            depth_path = os.path.join(self.image_depth_dir, f"{frame_id + 1}.png")
            robot_transform_path = os.path.join(self.robot_pose_dir, f"T_base2ee0{frame_id}.npz")

            # 读取图像和深度图
            color = o3d.io.read_image(image_path)
            depth = o3d.io.read_image(depth_path)
            robot_transform = np.linalg.inv(self.init_robot_transform) @ (
                        np.load(robot_transform_path)['arr_0'] @ self.camera_extrinsics)
            print(R.from_matrix(np.load(robot_transform_path)['arr_0'][:3, :3]).as_euler('xyz', degrees=True))
            # print(R.from_matrix(self.camera_extrinsics[:3, :3]).as_euler('xyz',degrees=True))
            # print(R.from_matrix(robot_transform[:3, :3]).as_euler('xyz',degrees=True))
            print(R.from_matrix(transformation_float_values[:3, :3]).as_euler('xyz', degrees=True))
            ####################################################
            robot_transform = transformation_float_values
            # 创建并返回Open3D RGBD图像对象
            rgbd_frame = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1000.0, depth_trunc=0.6, convert_rgb_to_intensity=False)
            if frame_id % 10 == 0:
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd_frame, self.camera_intrinsics)
                # 设置窗口属性并可视化点云
                pcd.transform(robot_transform)  # 翻转网格

                o3d.visualization.draw_geometries(
                    [pcd],  # 要显示的几何对象列表
                    window_name="Point Cloud",  # 窗口名称
                    width=1920,  # 窗口宽度
                    height=1080,  # 窗口高度
                    left=50,  # 窗口左上角横坐标
                    top=50,  # 窗口左上角纵坐标
                    point_show_normal=False,  # 是否显示点的法线
                    mesh_show_wireframe=False,  # 是否显示网格线框
                    mesh_show_back_face=False  # 是否显示背面的网格
                )
            self.volume.integrate(rgbd_frame, self.camera_intrinsics, extrinsic=robot_transform)  # 集成到 TSDF 体积
            # cv2.imshow('color', np.array(color))
            # transform = np.dot(transformation_float_values, self.transform_0)  # 计算累计变换
            # self.transform_0 = transform  # 更新当前变换
            # transforms_array.append(self.transform_0)  # 将当前数组添加到列表中
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            if frame_id % 10 == 0:
                pcd = self.volume.extract_point_cloud()
                # 设置窗口属性并可视化点云
                pcd.transform(robot_transform)  # 翻转网格

                o3d.visualization.draw_geometries(
                    [pcd],  # 要显示的几何对象列表
                    window_name="TSDF",  # 窗口名称
                    width=1920,  # 窗口宽度
                    height=1080,  # 窗口高度
                    left=50,  # 窗口左上角横坐标
                    top=50,  # 窗口左上角纵坐标
                    point_show_normal=False,  # 是否显示点的法线
                    mesh_show_wireframe=False,  # 是否显示网格线框
                    mesh_show_back_face=False  # 是否显示背面的网格
                )
            frame_id += 1

        # 循环结束
        # 将所有数组保存到一个.npz文件中
        np.save('config/transforms.npy', np.array(transforms_array))
        mesh = self.volume.extract_triangle_mesh()  # 提取三角网格

        print(mesh.compute_vertex_normals())  # 计算顶点法线
        mesh.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # 翻转网格

        o3d.visualization.draw_geometries([mesh])  # 显示网格
        o3d.io.write_triangle_mesh(
            'Mesh__%s__every_%sth_%sframes_gpu.ply' % ("file_name", "skip_N_frames", len("camera_poses")),
            mesh)  # 写入三角网格
        meshRead = o3d.io.read_triangle_mesh(
            'Mesh__%s__every_%sth_%sframes_gpu.ply' % ("file_name", "skip_N_frames", len("camera_poses")))  # 读取三角网格
        o3d.visualization.draw_geometries([meshRead])  # 显示网格
        self.executor.shutdown()


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
                self.pipeline_view.toggle_record.cis_on = False
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
    parser.add_argument('--camera-config',
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
