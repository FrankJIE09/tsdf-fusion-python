import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
import fusion
import open3d as o3d

def setup_realsense():
    # 创建 RealSense 管道
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # 启动管道
    profile = pipeline.start(config)
    return pipeline, profile

def get_frame_data(pipeline):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None, None
    # 将帧转换为numpy数组
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # 转换彩色图像到RGB
    return depth_image, color_image, depth_frame.profile.as_video_stream_profile().get_intrinsics()

if __name__ == "__main__":
    # 设置相机和TSDF融合体
    pipeline, profile = setup_realsense()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    tsdf_vol = fusion.TSDFVolume(vol_bnds=np.zeros((3, 2)), voxel_size=0.02)
    pcd_folder = "pcd_folder"
    os.makedirs(pcd_folder, exist_ok=True)

    # 收集和处理帧
    n_imgs = 1000  # 设定处理的图像数量
    print("Starting TSDF fusion of RealSense images...")
    start_time = time.time()

    try:
        for i in range(n_imgs):
            depth_image, color_image, intrinsics = get_frame_data(pipeline)
            if depth_image is None:
                continue
            # 转换深度图像到米
            depth_image = depth_image * depth_scale
            # 计算相机内参矩阵
            cam_intr = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                 [0, intrinsics.fy, intrinsics.ppy],
                                 [0, 0, 1]])
            # 使用假设的单位矩阵作为相机位姿
            cam_pose = np.eye(4)

            tsdf_vol.integrate(color_image, depth_image, cam_intr, cam_pose, obs_weight=1.)

            # 提取并保存每一帧的点云
            point_cloud = tsdf_vol.get_point_cloud()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
            if point_cloud.shape[1] > 3:
                pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:] / 255.0)
            pcd_filename = os.path.join(pcd_folder, f"frame-{i:06d}.pcd")
            o3d.io.write_point_cloud(pcd_filename, pcd)

    finally:
        pipeline.stop()

    elapsed_time = time.time() - start_time
    print(f"Finished processing {n_imgs} frames in {elapsed_time:.2f} seconds.")
