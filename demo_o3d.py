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


def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud.
    """
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):
    """Get corners of 3D camera view frustum of depth image
    """
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    view_frust_pts = np.array([
        (np.array([0, 0, 0, im_w, im_w]) - cam_intr[0, 2]) * np.array([0, max_depth, max_depth, max_depth, max_depth]) /
        cam_intr[0, 0],
        (np.array([0, 0, im_h, 0, im_h]) - cam_intr[1, 2]) * np.array([0, max_depth, max_depth, max_depth, max_depth]) /
        cam_intr[1, 1],
        np.array([0, max_depth, max_depth, max_depth, max_depth])
    ])
    view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
    return view_frust_pts


if __name__ == "__main__":
    print("Estimating voxel volume bounds...")  # 打印当前操作
    n_imgs = 1000  # 设置要处理的图像数量
    cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')  # 加载相机内参
    print("Initializing voxel volume...")  # 打印当前操作

    # 创建一个可扩展的 TSDF 体积实例
    # voxel_length：每个体素的长度，这里设置为0.02米
    # sdf_trunc：截断距离，用于控制TSDF的更新范围，这里设置为0.4米
    # color_type：设置为RGB8以支持颜色信息
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,
        sdf_trunc=.3,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    pcd_folder = "pcd_folder"
    os.makedirs(pcd_folder, exist_ok=True)
    clear_folder(pcd_folder)  # 清空点云文件夹

    t0_elapse = time.time()  # 记录起始时间
    for i in tqdm(range(n_imgs)):
        # print("Fusing frame %d/%d" % (i + 1, n_imgs))  # 打印当前处理的帧数

        color_image = cv2.cvtColor(cv2.imread("data/frame-%06d.color.jpg" % i), cv2.COLOR_BGR2RGB)  # 读取彩色图像并转换颜色空间
        depth_im = cv2.imread("data/frame-%06d.depth.png" % i, -1).astype(np.float32) # 读取深度图像并转换单位为米
        depth_im /= 1000.  # 将深度值从毫米转换为米

        depth_im[depth_im == 65.535] = 0  # 将无效深度值设为0

        color_raw = o3d.geometry.Image(color_image)
        depth_raw = o3d.geometry.Image(depth_im)

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(width=color_image.shape[1], height=color_image.shape[0],
                                 fx=cam_intr[0, 0], fy=cam_intr[1, 1], cx=cam_intr[0, 2], cy=cam_intr[1, 2])
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, convert_rgb_to_intensity=False)
        # 将RGBDImage转换为点云
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

        # 可视化点云
        o3d.visualization.draw_geometries([pcd])


        # 检查 RGB-D 图像是否有效
        cam_pose = np.loadtxt("data/frame-%06d.pose.txt" % i)  # 读取相机姿态

        volume.integrate(rgbd_image, intrinsic, cam_pose)

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
