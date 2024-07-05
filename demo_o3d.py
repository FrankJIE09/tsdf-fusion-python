import time
import cv2
import numpy as np
import os
import open3d as o3d
import shutil

from tqdm import tqdm


def load_intrinsics(file_path):
    """从文件加载相机内参，并返回NumPy数组形式。"""
    with open(file_path, 'r') as file:
        intrinsics = json.load(file)
        return np.array(intrinsics)


def clear_folder(folder_path):
    """清空指定文件夹中的所有内容。"""
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def get_view_frustum(depth_im, cam_intr, cam_pose):
    """计算相机视锥体并返回其顶点"""
    im_h, im_w = depth_im.shape

    max_depth = depth_im.max()
    # 计算像素坐标
    pix_x, pix_y = np.meshgrid(np.arange(im_w), np.arange(im_h))
    pix_x = pix_x.reshape(-1)
    pix_y = pix_y.reshape(-1)

    cam_pts = np.vstack((pix_x * depth_im.flatten(), pix_y * depth_im.flatten(), depth_im.flatten())).T
    cam_pts[:, 0] = (cam_pts[:, 0] - cam_intr[0, 2]) / cam_intr[0, 0]  # X in camera coordinates
    cam_pts[:, 1] = (cam_pts[:, 1] - cam_intr[1, 2]) / cam_intr[1, 1]  # Y in camera coordinates
    cam_pts[:, 2] = cam_pts[:, 2]  # Z in camera coordinates

    cam_pts_homog = np.hstack((cam_pts, np.ones((cam_pts.shape[0], 1))))
    world_pts_homog = cam_pose.dot(cam_pts_homog.T).T
    world_pts = world_pts_homog[:, :3] / world_pts_homog[:, 3:]

    return world_pts


if __name__ == "__main__":
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
        view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=0))  # 更新最小边界
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=0))  # 更新最大边界

    print("Initializing voxel volume...")  # 打印当前操作
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.02,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    pcd_folder = "pcd_folder"
    os.makedirs(pcd_folder, exist_ok=True)
    clear_folder(pcd_folder)  # 清空点云文件夹

    t0_elapse = time.time()  # 记录起始时间
    for i in tqdm(range(n_imgs)):
        print("Fusing frame %d/%d" % (i + 1, n_imgs))  # 打印当前处理的帧数

        color_image = cv2.cvtColor(cv2.imread("data/frame-%06d.color.jpg" % i), cv2.COLOR_BGR2RGB)  # 读取彩色图像并转换颜色空间
        depth_im = cv2.imread("data/frame-%06d.depth.png" % i, -1).astype(np.float32) / 1000.0  # 读取深度图像并转换单位为米
        depth_im[depth_im == 65.535] = 0  # 将无效深度值设为0
        cam_pose = np.loadtxt("data/frame-%06d.pose.txt" % i)  # 读取相机姿态

        color_raw = o3d.geometry.Image(color_image)
        depth_raw = o3d.geometry.Image(depth_im)

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(width=color_image.shape[1], height=color_image.shape[0],
                                 fx=cam_intr[0, 0], fy=cam_intr[1, 1], cx=cam_intr[0, 2], cy=cam_intr[1, 2])

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, convert_rgb_to_intensity=False)

        volume.integrate(rgbd_image, intrinsic, np.linalg.inv(cam_pose))

        cv2.imshow('Color Image', color_image)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_im, alpha=255 / depth_im.max()), cv2.COLORMAP_JET)
        cv2.imshow('Depth Image', depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if i % 10 == 0:
            point_cloud = volume.extract_point_cloud()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.asarray(point_cloud.points))  # 更新点云坐标
            if np.asarray(point_cloud.colors).shape[1] == 3:
                pcd.colors = o3d.utility.Vector3dVector(np.asarray(point_cloud.colors))  # 更新点云颜色
            pcd_filename = os.path.join(pcd_folder, f"frame-{i:06d}.pcd")
            o3d.io.write_point_cloud(pcd_filename, pcd)

    fps = n_imgs / (time.time() - t0_elapse)  # 计算平均帧率
    print("Average FPS: {:.2f}".format(fps))  # 打印平均帧率

    print("Saving mesh to mesh.ply...")  # 打印当前操作
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh("mesh.ply", mesh)

    print("Saving point cloud to pc.ply...")  # 打印当前操作
    point_cloud = volume.extract_point_cloud()
    o3d.io.write_point_cloud("pc.ply", point_cloud)

    print("Mesh and point cloud saved.")

    pcd = o3d.io.read_point_cloud("pc.ply")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Point Cloud",
        width=1920,
        height=1080,
        left=50,
        top=50,
        point_show_normal=False,
        mesh_show_wireframe=False,
        mesh_show_back_face=False
    )

    cv2.destroyAllWindows()
