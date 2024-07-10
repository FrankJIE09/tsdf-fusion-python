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

    @property
    def max_points(self):
        """计算摄像头或RGBD视频分辨率下的单帧最大点数。"""
        return self.rgbd_metadata.width * self.rgbd_metadata.height

    @property
    def vfov(self):
        """获取摄像头或RGBD视频的垂直视场角。"""
        return np.rad2deg(2 * np.arctan(self.intrinsic_matrix[1, 2].item() /
                                        self.intrinsic_matrix[1, 1].item()))

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

            if frame_id == 0:
                frame_id += 1
                self.source_down = self.pcd_frame
                continue
            target_down = self.pcd_frame
            result_ransac = o3d.t.pipelines.registration.icp(self.source_down, target_down,
                                                             # init_source_to_target=camera_poses_cpu[i+1].pose,
                                                             estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
                                                             max_correspondence_distance=self.voxel_size * 1.5,
                                                             voxel_size=self.voxel_size,
                                                             criteria=o3d.t.pipelines.registration.ICPConvergenceCriteria(
                                                                 max_iteration=1000))
            self.source_down = self.pcd_frame

            transform = result_ransac.transformation  # 获取变换矩阵

            transformation_float_values = np.eye(4)
            # 遍历张量的每个元素，并将其转换为 float
            for m in range(transform.shape[0]):
                for n in range(transform.shape[1]):
                    transformation_float_values[m][n] = transform[m][n].item()
            transform = np.dot(transformation_float_values, self.transform_0)  # 计算累计变换
            self.transform_0 = transform  # 更新当前变换

            print(self.pcd_frame)
            color_image_ = o3d.t.geometry.Image.to_legacy(self.rgbd_frame.color)
            color_depth_ = o3d.t.geometry.Image.to_legacy(self.rgbd_frame.depth)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image_, color_depth_,
                depth_trunc=0.5,  # 深度截断
                convert_rgb_to_intensity=False)  # 创建 RGBD 图像
            self.volume.integrate(rgbd, self.rgbd_metadata.intrinsics, self.transform_0)  # 集成到 TSDF 体积
            # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            #     rgbd, intrinsic=self.rgbd_metadata.intrinsics
            # )
            # o3d.visualization.draw_geometries([pcd])4
            cv2.imshow('color',np.array(color_image_))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_id += 1
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
        log.debug(f"create_from_depth_image() errors = {pcd_errors}")

    def toggle_record(self):
        if self.camera is not None:
            if self.flag_record and not self.recording:
                self.camera.resume_record()
                self.recording = True
            elif not self.flag_record and self.recording:
                self.camera.pause_record()
                self.recording = False

    def save_pcd(self):
        """保存当前点云。"""
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"{self.rgbd_metadata.serial_number}_pcd_{now}.ply"
        # 转换颜色为 uint8 以确保兼容性
        self.pcd_frame.point.colors = (self.pcd_frame.point.colors * 255).to(
            o3d.core.Dtype.UInt8)
        self.executor.submit(o3d.t.io.write_point_cloud,
                             filename,
                             self.pcd_frame,
                             write_ascii=False,
                             compressed=True,
                             print_progress=False)
        self.status_message = f"Saving point cloud to {filename}."

    def save_rgbd(self):
        """保存当前RGBD图像对。"""
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"{self.rgbd_metadata.serial_number}_color_{now}.jpg"
        self.executor.submit(o3d.t.io.write_image, filename,
                             self.rgbd_frame.color)
        filename = f"{self.rgbd_metadata.serial_number}_depth_{now}.png"
        self.executor.submit(o3d.t.io.write_image, filename,
                             self.rgbd_frame.depth)
        self.status_message = (
            f"Saving RGBD images to {filename[:-3]}.{{jpg,png}}.")


# GUI 控制显示和用户界面的类。所有方法必须在主线程中运行。
class PipelineView:
    """控制显示和用户界面。所有方法必须在主线程中运行。"""

    def __init__(self, vfov=60, max_pcd_vertices=1 << 20, **callbacks):
        """初始化。
        Args:
            vfov (float): 3D场景的垂直视场角。
            max_pcd_vertices (int): 分配内存的最大点云顶点数。
            callbacks (dict of kwargs): 控制器提供的各种操作的回调。
        """

        self.vfov = vfov
        self.max_pcd_vertices = max_pcd_vertices

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(
            "Open3D || Online RGBD Video Processing", 1280, 960)
        # 窗口布局调整时调用
        self.window.set_on_layout(self.on_layout)
        self.window.set_on_close(callbacks['on_window_close'])

        self.pcd_material = o3d.visualization.rendering.MaterialRecord()
        self.pcd_material.shader = "defaultLit"
        # 设置每个3D点显示的像素数，考虑高DPI缩放
        self.pcd_material.point_size = int(4 * self.window.scaling)

        # 3D场景设置
        self.pcdview = gui.SceneWidget()
        self.window.add_child(self.pcdview)
        self.pcdview.enable_scene_caching(
            True)  # 使UI响应更快
        self.pcdview.scene = rendering.Open3DScene(self.window.renderer)
        self.pcdview.scene.set_background([1, 1, 1, 1])  # 白色背景
        self.pcdview.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, [0, -6, 0])
        # 设置点云边界，依赖于传感器范围
        self.pcd_bounds = o3d.geometry.AxisAlignedBoundingBox([-3, -3, 0],
                                                              [3, 3, 6])
        self.camera_view()  # 初始从摄像头视角查看
        em = self.window.theme.font_size

        # 选项面板
        self.panel = gui.Vert(em, gui.Margins(em, em, em, em))
        self.panel.preferred_width = int(360 * self.window.scaling)
        self.window.add_child(self.panel)
        toggles = gui.Horiz(em)
        self.panel.add_child(toggles)

        toggle_capture = gui.ToggleSwitch("Capture / Play")
        toggle_capture.is_on = False
        toggle_capture.set_on_clicked(
            callbacks['on_toggle_capture'])  # 回调
        toggles.add_child(toggle_capture)

        self.flag_normals = False
        self.toggle_normals = gui.ToggleSwitch("Colors / Normals")
        self.toggle_normals.is_on = False
        self.toggle_normals.set_on_clicked(
            callbacks['on_toggle_normals'])  # 回调
        toggles.add_child(self.toggle_normals)

        view_buttons = gui.Horiz(em)
        self.panel.add_child(view_buttons)
        view_buttons.add_stretch()  # 用于居中
        camera_view = gui.Button("Camera view")
        camera_view.set_on_clicked(self.camera_view)  # 回调
        view_buttons.add_child(camera_view)
        birds_eye_view = gui.Button("Bird's eye view")
        birds_eye_view.set_on_clicked(self.birds_eye_view)  # 回调
        view_buttons.add_child(birds_eye_view)
        view_buttons.add_stretch()  # 用于居中

        save_toggle = gui.Horiz(em)
        self.panel.add_child(save_toggle)
        save_toggle.add_child(gui.Label("Record / Save"))
        self.toggle_record = None
        if callbacks['on_toggle_record'] is not None:
            save_toggle.add_fixed(1.5 * em)
            self.toggle_record = gui.ToggleSwitch("Video")
            self.toggle_record.is_on = False
            self.toggle_record.set_on_clicked(callbacks['on_toggle_record'])
            save_toggle.add_child(self.toggle_record)

        save_buttons = gui.Horiz(em)
        self.panel.add_child(save_buttons)
        save_buttons.add_stretch()  # 用于居中
        save_pcd = gui.Button("Save Point cloud")
        save_pcd.set_on_clicked(callbacks['on_save_pcd'])
        save_buttons.add_child(save_pcd)
        save_rgbd = gui.Button("Save RGBD frame")
        save_rgbd.set_on_clicked(callbacks['on_save_rgbd'])
        save_buttons.add_child(save_rgbd)
        save_buttons.add_stretch()  # 用于居中

        self.video_size = (int(240 * self.window.scaling),
                           int(320 * self.window.scaling), 3)
        self.show_color = gui.CollapsableVert("Color image")
        self.show_color.set_is_open(False)
        self.panel.add_child(self.show_color)
        self.color_video = gui.ImageWidget(
            o3d.geometry.Image(np.zeros(self.video_size, dtype=np.uint8)))
        self.show_color.add_child(self.color_video)
        self.show_depth = gui.CollapsableVert("Depth image")
        self.show_depth.set_is_open(False)
        self.panel.add_child(self.show_depth)
        self.depth_video = gui.ImageWidget(
            o3d.geometry.Image(np.zeros(self.video_size, dtype=np.uint8)))
        self.show_depth.add_child(self.depth_video)

        self.status_message = gui.Label("")
        self.panel.add_child(self.status_message)

        self.flag_exit = False
        self.flag_gui_init = False

    def update(self, frame_elements):
        """使用点云和图像更新可视化。必须在主线程中运行，因为这涉及到GUI调用。

        Args:
            frame_elements: dict {element_type: geometry element}.
                字典，键为元素类型，值为要在GUI中更新的几何元素：
                    'pcd': 点云,
                    'color': rgb图像（3通道，uint8）,
                    'depth': 深度图像（uint8）,
                    'status_message': 消息
        """
        if not self.flag_gui_init:
            # 设置虚拟点云以分配图形内存
            dummy_pcd = o3d.t.geometry.PointCloud({
                'positions':
                    o3d.core.Tensor.zeros((self.max_pcd_vertices, 3),
                                          o3d.core.Dtype.Float32),
                'colors':
                    o3d.core.Tensor.zeros((self.max_pcd_vertices, 3),
                                          o3d.core.Dtype.Float32),
                'normals':
                    o3d.core.Tensor.zeros((self.max_pcd_vertices, 3),
                                          o3d.core.Dtype.Float32)
            })
            if self.pcdview.scene.has_geometry('pcd'):
                self.pcdview.scene.remove_geometry('pcd')

            self.pcd_material.shader = "normals" if self.flag_normals else "defaultLit"
            self.pcdview.scene.add_geometry('pcd', dummy_pcd, self.pcd_material)
            self.flag_gui_init = True

        # (ssheorey) 切换到update_geometry()在 #3452 修复后
        if os.name == 'nt':
            self.pcdview.scene.remove_geometry('pcd')
            self.pcdview.scene.add_geometry('pcd', frame_elements['pcd'],
                                            self.pcd_material)
        else:
            update_flags = (rendering.Scene.UPDATE_POINTS_FLAG |
                            rendering.Scene.UPDATE_COLORS_FLAG |
                            (rendering.Scene.UPDATE_NORMALS_FLAG
                             if self.flag_normals else 0))
            self.pcdview.scene.scene.update_geometry('pcd',
                                                     frame_elements['pcd'],
                                                     update_flags)

        # 更新颜色和深度图像
        # (ssheorey) 删除CPU转换，在我们有CUDA -> OpenGL桥后
        if self.show_color.get_is_open() and 'color' in frame_elements:
            sampling_ratio = self.video_size[1] / frame_elements['color'].columns
            self.color_video.update_image(
                frame_elements['color'].resize(sampling_ratio).cpu())
        if self.show_depth.get_is_open() and 'depth' in frame_elements:
            sampling_ratio = self.video_size[1] / frame_elements['depth'].columns
            self.depth_video.update_image(
                frame_elements['depth'].resize(sampling_ratio).cpu())

        if 'status_message' in frame_elements:
            self.status_message.text = frame_elements["status_message"]

        self.pcdview.force_redraw()

    def camera_view(self):
        """回调函数，重置点云视图到摄像头"""
        self.pcdview.setup_camera(self.vfov, self.pcd_bounds, [0, 0, 0])
        # 从摄像头位置 [0, 0, 0] 查看点 [0, 0, 1]，Y轴朝向 [0, -1, 0]
        self.pcdview.scene.camera.look_at([0, 0, 1], [0, 0, 0], [0, -1, 0])

    def birds_eye_view(self):
        """回调函数，重置点云视图到鸟瞰图（顶视图）"""
        self.pcdview.setup_camera(self.vfov, self.pcd_bounds, [0, 0, 0])
        self.pcdview.scene.camera.look_at([0, 0, 1.5], [0, 3, 1.5], [0, -1, 0])

    def on_layout(self, layout_context):
        # on_layout 回调应设置每个子元素的位置和大小。回调完成后，窗口将布局子孙元素。
        """窗口初始化/调整大小时的回调函数"""
        frame = self.window.content_rect
        self.pcdview.frame = frame
        panel_size = self.panel.calc_preferred_size(layout_context,
                                                    self.panel.Constraints())
        self.panel.frame = gui.Rect(frame.get_right() - panel_size.width,
                                    frame.y, panel_size.width,
                                    panel_size.height)


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
