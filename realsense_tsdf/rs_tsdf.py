import numpy as np  # 导入NumPy库，用于数学和矩阵运算
import cv2  # 导入OpenCV库，用于图像处理
import json  # 导入json库，用于处理JSON数据格式
import os  # 导入os库，用于操作文件系统

from tqdm import tqdm

import fusion  # 导入自定义的fusion库，可能用于处理三维数据融合
import open3d as o3d  # 导入Open3D库，用于三维数据处理和可视化
import shutil  # 导入shutil库，用于文件和文件夹的高级操作


def load_intrinsics(file_path):
    """从文件加载相机内参，并返回NumPy数组形式。"""
    with open(file_path, 'r') as file:
        intrinsics = json.load(file)  # 读取JSON格式的相机内参
        return np.array(intrinsics)  # 转换为NumPy数组并返回


def clear_folder(folder_path):
    """清空指定文件夹中的所有内容。"""
    for item in os.listdir(folder_path):  # 遍历文件夹中的每个项目
        item_path = os.path.join(folder_path, item)  # 获取项目的完整路径
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)  # 如果是文件或链接，则删除
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # 如果是文件夹，则递归删除


def main():
    save_folder = "captured_images"  # 定义保存图像的文件夹路径
    # 统计以"color_"开始的文件数量
    frame_count = len([name for name in os.listdir(save_folder) if name.startswith("color_")])

    # 初始化体素体积的边界
    vol_bnds = np.zeros((3, 2))  # 创建一个3x2的零矩阵，用于存储XYZ的最小和最大边界

    for i in range(frame_count):
        # 加载和处理每一帧的图像和数据
        cam_pose_path = os.path.join(save_folder, f"pose_{i}.txt")
        cam_pose = np.loadtxt(cam_pose_path)  # 加载相机姿态文件

        depth_img_path = os.path.join(save_folder, f"depth_{i}.png")  # 构建深度图像文件路径
        depth_im = cv2.imread(depth_img_path).astype(float) / 1000  # 读取深度图像

        intrinsics_path = os.path.join(save_folder, f"intrinsics_{i}.json")  # 构建内参文件路径
        cam_intr = load_intrinsics(intrinsics_path)  # 加载相机内参

        # 计算视锥并更新体素体积的边界
        view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)  # 计算视锥点
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))  # 更新边界最小值
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))  # 更新边界最大值

    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.005, sdf_trunc=0.1)  # 创建TSDF体素体积对象，体素大小为2cm
    pcd_folder = "rs_pcd_folder"  # 定义保存点云的文件夹
    os.makedirs(pcd_folder, exist_ok=True)  # 如果文件夹不存在则创建
    clear_folder(pcd_folder)  # 清空点云文件夹

    for i in tqdm(range(frame_count)):
        # 为每帧生成和保存点云
        color_img_path = os.path.join(save_folder, f"color_{i}.png")
        depth_img_path = os.path.join(save_folder, f"depth_{i}.png")
        intrinsics_path = os.path.join(save_folder, f"intrinsics_{i}.json")

        color_image = cv2.imread(color_img_path)  # 重新读取彩色图像
        depth_image = cv2.imread(depth_img_path,-1).astype(float) / 1000  # 重新读取深度图像并转换单位为米
        cam_intr = load_intrinsics(intrinsics_path)  # 重新加载内参

        cam_pose_path = os.path.join(save_folder, f"pose_{i}.txt")
        cam_pose = np.loadtxt(cam_pose_path)  # 加载相机姿态文件

        cv2.imshow('Color Image', color_image)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=255 / depth_im.max()), cv2.COLORMAP_JET)
        cv2.imshow('Depth Image', depth_colormap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # 融合观测结果到体素体积
        tsdf_vol.integrate(color_image, depth_image, cam_intr, cam_pose, obs_weight=1.)
    print("Extracting mesh...")  # 开始提取网格
    mesh = tsdf_vol.get_mesh()  # 从TSDF体积提取网格
    mesh_path = os.path.join(save_folder, "mesh.ply")  # 构建网格文件路径
    fusion.meshwrite(mesh_path, *mesh)  # 保存网格文件

    point_cloud = tsdf_vol.get_point_cloud()  # 获取点云
    fusion.pcwrite("pc.ply", point_cloud)  # 将点云保存到文件
    print(f"Mesh saved to {mesh_path}")  # 打印网格保存路径
    pcd = o3d.io.read_point_cloud("captured_images/mesh.ply")

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


if __name__ == "__main__":
    main()  # 执行主函数
