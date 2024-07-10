# License: Apache 2.0. See LICENSE file in root directory.
# 许可证：Apache 2.0。请参阅根目录中的LICENSE文件。

# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.
# 版权所有(c) 2015-2017 Intel Corporation。保留所有权利。

"""
OpenCV and Numpy Point cloud Software Renderer
OpenCV和Numpy点云软件渲染器

This sample is mostly for demonstration and educational purposes.
这个示例主要用于演示和教育目的。

It really doesn't offer the quality or performance that can be
achieved with hardware acceleration.
它并没有提供硬件加速所能达到的质量或性能。

Usage:
用法:
------
Mouse:
鼠标：
    Drag with left button to rotate around pivot (thick small axes),
    使用左键拖动以围绕枢轴旋转（粗小轴），
    with right button to translate and the wheel to zoom.
    使用右键平移，使用滚轮缩放。

Keyboard:
键盘：
    [p]     Pause
    [p]     暂停
    [r]     Reset View
    [r]     重置视图
    [d]     Cycle through decimation values
    [d]     循环切换降采样值
    [z]     Toggle point scaling
    [z]     切换点缩放
    [c]     Toggle color source
    [c]     切换颜色源
    [s]     Save PNG (./out.png)
    [s]     保存PNG（./out.png）
    [e]     Export points to ply (./out.ply)
    [e]     导出点云到ply（./out.ply）
    [q\ESC] Quit
    [q\ESC] 退出
"""

import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d


class AppState:
    # 初始化AppState类
    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        # 设置初始视角
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        # 设置初始平移
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        # 设置初始鼠标状态
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        # 设置初始其他状态
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    # 重置视图
    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    # 计算旋转矩阵
    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    # 计算枢轴位置
    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


state = AppState()

# 配置深度和颜色流
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

# 检查是否找到RGB摄像头
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# 启用深度和颜色流
config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

# 开始流媒体
pipeline.start(config)

# 获取流配置和相机内参
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# 处理块
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
colorizer = rs.colorizer()


# 定义鼠标回调函数
def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True

    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False

    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True

    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False

    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True

    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:

        h, w = out.shape[:2]
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2

        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        elif state.mouse_btns[2]:
            dz = math.sqrt(dx ** 2 + dy ** 2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz

    state.prev_mouse = (x, y)


# 设置窗口和鼠标回调
cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(state.WIN_NAME, w, h)
cv2.setMouseCallback(state.WIN_NAME, mouse_cb)


# 将3D向量投影到2D
def project(v):
    """project 3d vector array to 2d"""
    h, w = out.shape[:2]
    view_aspect = float(h) / w

    # 忽略无效深度的除以零错误
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
               (w * view_aspect, h) + (w / 2.0, h / 2.0)

    # 近裁剪
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj


# 应用视图变换
def view(v):
    """apply view transformation on vector array"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


# 绘制3D线条
def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
    """draw a 3d line from pt1 to pt2"""
    p0 = project(pt1.reshape(-1, 3))[0]
    p1 = project(pt2.reshape(-1, 3))[0]
    if np.isnan(p0).any() or np.isnan(p1).any():
        return
    p0 = tuple(p0.astype(int))
    p1 = tuple(p1.astype(int))
    rect = (0, 0, out.shape[1], out.shape[0])
    inside, p0, p1 = cv2.clipLine(rect, p0, p1)
    if inside:
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


# 在xz平面绘制网格
def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
    """draw a grid on xz plane"""
    pos = np.array(pos)
    s = size / float(n)
    s2 = 0.5 * size
    for i in range(0, n + 1):
        x = -s2 + i * s
        line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
               view(pos + np.dot((x, 0, s2), rotation)), color)
    for i in range(0, n + 1):
        z = -s2 + i * s
        line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
               view(pos + np.dot((s2, 0, z), rotation)), color)


# 绘制3D坐标轴
def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
    """draw 3d axes"""
    line3d(out, pos, pos +
           np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
    line3d(out, pos, pos +
           np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
    line3d(out, pos, pos +
           np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


# 绘制相机视锥
def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
    """draw camera's frustum"""
    orig = view([0, 0, 0])
    w, h = intrinsics.width, intrinsics.height

    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            line3d(out, orig, view(p), color)
            return p

        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        bottom_left = get_point(0, h)

        line3d(out, view(top_left), view(top_right), color)
        line3d(out, view(top_right), view(bottom_right), color)
        line3d(out, view(bottom_right), view(bottom_left), color)
        line3d(out, view(bottom_left), view(top_left), color)


def create_colored_point_cloud(verts, texcoords, color_image):
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()

    # 设置点云的顶点
    pcd.points = o3d.utility.Vector3dVector(verts)

    # 从纹理坐标提取颜色信息
    # 首先获取纹理坐标对应的图像坐标
    h, w, _ = color_image.shape
    texcoords_index = (texcoords * np.array([w, h])).astype(np.int32)
    texcoords_index[:, 0] = np.clip(texcoords_index[:, 0], 0, w - 1)
    texcoords_index[:, 1] = np.clip(texcoords_index[:, 1], 0, h - 1)

    # 从颜色图像中获取颜色值
    colors = color_image[texcoords_index[:, 1], texcoords_index[:, 0], :]  # 注意纹理坐标对应 (u, v) 映射到 (列, 行)
    colors = colors / 255.0  # 归一化到 [0, 1] 范围

    # 设置点云颜色
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


# 绘制点云
def pointcloud(out, verts, texcoords, color, painter=True):
    """draw point cloud with optional painter's algorithm"""
    if painter:
        # 使用画家算法，从后往前排序点
        # 如果使用画家算法，则从后往前排序顶点以确保正确的绘制顺序

        # 按z（视图空间）反向排序的索引
        v = view(verts)
        # 将顶点从世界坐标转换到视图坐标
        s = v[:, 2].argsort()[::-1]
        # 根据z轴（深度）从后往前排序，并获取排序后的索引
        proj = project(v[s])
        # 根据排序后的顶点进行投影，得到2D图像坐标
    else:
        proj = project(view(verts))
        # 如果不使用画家算法，则直接将顶点投影到2D图像坐标

    if state.scale:
        proj *= 0.5 ** state.decimate
        # 如果启用了缩放，根据缩减比例缩放投影坐标

    h, w = out.shape[:2]
    # 获取输出图像的高度和宽度

    # proj现在包含2D图像坐标
    j, i = proj.astype(np.uint32).T
    # 将投影坐标转换为整数，并分别获取x和y坐标

    # 创建掩码以忽略越界索引
    im = (i >= 0) & (i < h)
    # 创建一个掩码，检查y坐标是否在图像高度范围内
    jm = (j >= 0) & (j < w)
    # 创建一个掩码，检查x坐标是否在图像宽度范围内
    m = im & jm
    # 组合掩码，只有在x和y坐标都在范围内时才为True

    cw, ch = color.shape[:2][::-1]
    # 获取颜色图像的宽度和高度

    if painter:
        # 使用与上面相同的索引排序texcoord
        # texcoords是[0..1]，相对于左上角像素，
        # 乘以尺寸并加0.5以居中
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
        # 根据排序后的索引对纹理坐标进行排序，并将其转换为图像坐标
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
        # 如果不使用画家算法，直接将纹理坐标转换为图像坐标

    # 将texcoords裁剪到图像
    np.clip(u, 0, ch - 1, out=u)
    # 裁剪纹理坐标u，使其在图像宽度范围内
    np.clip(v, 0, cw - 1, out=v)
    # 裁剪纹理坐标v，使其在图像高度范围内

    # 执行uv映射
    out[i[m], j[m]] = color[u[m], v[m]]
    # 使用纹理坐标在颜色图像中查找颜色值，并将其赋给输出图像的对应像素


# 创建空的图像缓冲区
out = np.empty((h, w, 3), dtype=np.uint8)

while True:
    # 获取相机数据
    if not state.paused:
        # 等待一对一致的帧：深度和颜色
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_frame = decimate.process(depth_frame)

        # 获取新的内参（可能会被降采样改变）
        depth_intrinsics = rs.video_stream_profile(
            depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # # Create RGBD image
        # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        #     o3d.geometry.Image(color_image),
        #     o3d.geometry.Image(depth_image),
        #     depth_scale=1000.0,
        #     depth_trunc=3.0,
        #     convert_rgb_to_intensity=False
        # )
        #
        # # Create point cloud from RGBD
        # intrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.depth)).get_intrinsics()
        # pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
        # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)
        #
        # # Visualize the point cloud
        # o3d.visualization.draw_geometries([pcd])

        # 将深度图像数据转换为伪彩色图像，以便进行可视化显示
        depth_colormap = np.asanyarray(
            colorizer.colorize(depth_frame).get_data())

        # 根据当前状态选择颜色源
        if state.color:
            # 如果选择显示彩色图像
            mapped_frame, color_source = color_frame, color_image
        else:
            # 如果选择显示深度图像的伪彩色图
            mapped_frame, color_source = depth_frame, depth_colormap
        # 计算点云数据
        # 通过传入深度帧数据，生成点云对象（点的集合）
        points = pc.calculate(depth_frame)
        pc.map_to(mapped_frame)

        # 提取顶点和纹理坐标
        v, t = points.get_vertices(), points.get_texture_coordinates()

        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

    # 渲染
    now = time.time()
    # 获取当前时间，记录开始处理的时间戳

    out.fill(0)
    # 将输出图像数组 'out' 填充为全黑（所有像素值设为0）

    # grid(out, (0, 0.5, 1), size=1, n=10)
    # 在输出图像 'out' 上绘制一个网格，颜色为 (0, 0.5, 1)，大小为1，网格行列数为10

    # frustum(out, depth_intrinsics)
    # 在输出图像 'out' 上绘制一个表示相机视锥体的图形，使用深度相机的内参 'depth_intrinsics'

    # axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)
    # 在输出图像 'out' 上绘制坐标轴，原点为 (0, 0, 0)，使用 'state.rotation' 进行旋转，坐标轴大小为0.1，厚度为1

    # 如果 state.scale 为 False 或者输出图像 'out' 的尺寸等于 (h, w)，直接绘制点云
    # 否则，在临时图像 'tmp' 中绘制点云，然后将 'tmp' 的大小调整为 'out' 的大小，使用最近邻插值
    # 将临时图像 'tmp' 中非黑色部分覆盖到输出图像 'out' 中
    if not state.scale or out.shape[:2] == (h, w):
        pointcloud(out, verts, texcoords, color_source, painter=True)
    else:
        tmp = np.zeros((h, w, 3), dtype=np.uint8)
        pointcloud(tmp, verts, texcoords, color_source, painter=True)
        tmp = cv2.resize(
            tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        np.putmask(out, tmp > 0, tmp)

    # if any(state.mouse_btns):
    #     axes(out, view(state.pivot), state.rotation, thickness=4)
    # 如果有任何鼠标按钮被按下，在输出图像 'out' 上绘制坐标轴，原点为 'state.pivot'，使用 'state.rotation' 进行旋转，厚度为4

    dt = time.time() - now
    # 计算处理时间，记录处理一帧所用的时间

    cv2.setWindowTitle(
        state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
                        (w, h, 1.0 / dt, dt * 1000, "PAUSED" if state.paused else ""))
    # 设置窗口标题，显示分辨率 (w x h)，帧率 (FPS)，每帧处理时间 (ms)，以及状态（是否暂停）

    cv2.imshow(state.WIN_NAME, out)
    # 显示输出图像 'out' 到窗口中，窗口名称为 'state.WIN_NAME'

    key = cv2.waitKey(1)
    # 等待键盘事件，延迟1毫秒，获取按键的 ASCII 码值
    continue
    if key == ord("r"):
        state.reset()

    if key == ord("p"):
        state.paused ^= True

    if key == ord("d"):
        state.decimate = (state.decimate + 1) % 3
        decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

    if key == ord("z"):
        state.scale ^= True

    if key == ord("c"):
        state.color ^= True

    if key == ord("s"):
        cv2.imwrite('./out.png', out)

    if key == ord("e"):
        points.export_to_ply('./out.ply', mapped_frame)

    if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
        break

# 停止流媒体
pipeline.stop()
