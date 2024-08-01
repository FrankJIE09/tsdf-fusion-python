import pyrealsense2 as rs
import numpy as np
import trimesh

# 配置RealSense摄像头
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# 获取一帧点云数据
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()
if not depth_frame or not color_frame:
    raise ValueError("无法获取帧数据")

# 将深度帧转换为点云
pc = rs.pointcloud()
pc.map_to(color_frame)
points = pc.calculate(depth_frame)

# 获取点云的顶点和颜色数据
vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
colors = np.asanyarray(color_frame.get_data()).reshape((480, 640, 3))

# 获取每个顶点的颜色
color_list = []
for vert in tex:
    u = int(vert[0] * color_frame.width)
    v = int(vert[1] * color_frame.height)
    color_list.append(colors[v, u])

color_list = np.array(color_list)

# 创建一个Trimesh对象
mesh = trimesh.Trimesh(vertices=vtx, vertex_colors=color_list)

# 保存为.obj文件
mesh.export('mymesh.obj')

# 停止RealSense摄像头
pipeline.stop()
