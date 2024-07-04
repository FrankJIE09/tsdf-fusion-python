"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time
import cv2
import numpy as np
import fusion
from skimage import measure

if __name__ == "__main__":
    # ======================================================================================================== #
    # 计算数据集中所有相机视锥的凸包在世界坐标系中的3D边界
    # ======================================================================================================== #
    print("Estimating voxel volume bounds...")  # 打印当前操作
    n_imgs = 1000  # 设置要处理的图像数量
    cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')  # 加载相机内参
    vol_bnds = np.zeros((3, 2))  # 初始化体素体积的边界

    for i in range(n_imgs):
        # 读取深度图像和相机姿态
        depth_im = cv2.imread("data/frame-%06d.depth.png" % i, -1).astype(float)  # 读取深度图像
        depth_im /= 1000.  # 将深度值从毫米转换为米
        depth_im[depth_im == 65.535] = 0  # 将无效深度值设为0（特定于7-scenes数据集）
        cam_pose = np.loadtxt("data/frame-%06d.pose.txt" % i)  # 读取4x4刚体变换矩阵

        # 计算相机视锥并扩展凸包
        view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))  # 更新最小边界
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))  # 更新最大边界
    # ======================================================================================================== #

    # ======================================================================================================== #
    # 融合RGB-D图像
    # ======================================================================================================== #
    # 初始化体素体积
    print("Initializing voxel volume...")  # 打印当前操作
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)  # 创建TSDF体素体积对象，体素大小为2cm

    # 遍历所有的RGB-D图像并将它们融合在一起
    t0_elapse = time.time()  # 记录起始时间
    for i in range(n_imgs):
        print("Fusing frame %d/%d" % (i + 1, n_imgs))  # 打印当前处理的帧数

        # 读取RGB-D图像和相机姿态
        color_image = cv2.cvtColor(cv2.imread("data/frame-%06d.color.jpg" % i), cv2.COLOR_BGR2RGB)  # 读取彩色图像并转换颜色空间
        depth_im = cv2.imread("data/frame-%06d.depth.png" % i, -1).astype(float)  # 读取深度图像
        depth_im /= 1000.  # 将深度值从毫米转换为米
        depth_im[depth_im == 65.535] = 0  # 将无效深度值设为0
        cam_pose = np.loadtxt("data/frame-%06d.pose.txt" % i)  # 读取相机姿态

        # 将观测结果融合到体素体积中（假设颜色与深度对齐）
        tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)  # 融合当前帧

    fps = n_imgs / (time.time() - t0_elapse)  # 计算平均帧率
    print("Average FPS: {:.2f}".format(fps))  # 打印平均帧率

    # 从体素体积中提取网格并保存到磁盘（可以用Meshlab查看）
    print("Saving mesh to mesh.ply...")  # 打印当前操作
    verts, faces, norms, colors = tsdf_vol.get_mesh()  # 获取网格
    fusion.meshwrite("mesh.ply", verts, faces, norms, colors)  # 将网格保存到文件

    # 从体素体积中提取点云并保存到磁盘（可以用Meshlab查看）
    print("Saving point cloud to pc.ply...")  # 打印当前操作
    point_cloud = tsdf_vol.get_point_cloud()  # 获取点云
    fusion.pcwrite("pc.ply", point_cloud)  # 将点云保存到文件
