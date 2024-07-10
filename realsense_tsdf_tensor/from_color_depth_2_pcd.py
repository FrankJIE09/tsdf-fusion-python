import open3d as o3d
import numpy as np
import cv2


def create_point_cloud_from_image(image_path, depth_path):
    # 读取图像和深度图
    color_raw = o3d.io.read_image(image_path)
    depth_raw = o3d.io.read_image(depth_path)

    # 创建RGBD图像
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)

    # 创建相机内参（这里需要根据你的相机进行调整）
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    # 从RGBD图像生成点云
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsics)

    # 转换坐标系（如果需要的话）
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # 保存点云
    o3d.io.write_point_cloud("../realsense_tsdf/output.pcd", pcd)

    return pcd


# 示例使用的图片路径
image_path = "923322070420_color_2024-07-10_15-41-19.jpg"
depth_path = "923322070420_depth_2024-07-10_15-41-19.png"

# 生成点云
pcd = create_point_cloud_from_image(image_path, depth_path)

### 步骤3: 显示点云

# 使用Open3D的可视化功能来显示点云
o3d.visualization.draw_geometries([pcd])
