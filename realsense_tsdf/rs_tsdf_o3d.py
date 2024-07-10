import open3d as o3d
import numpy as np
import cv2
import json
import os
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


def main():
    save_folder = "captured_images"
    frame_count = len([name for name in os.listdir(save_folder) if name.startswith("color_")])

    # 初始化体素体积
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.02,
        sdf_trunc=1,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    pcd_folder = "rs_pcd_folder"
    os.makedirs(pcd_folder, exist_ok=True)
    clear_folder(pcd_folder)

    for i in tqdm(range(frame_count)):
        color_img_path = os.path.join(save_folder, f"color_{i}.png")
        depth_img_path = os.path.join(save_folder, f"depth_{i}.png")
        intrinsics_path = os.path.join(save_folder, f"intrinsics_{i}.json")
        cam_pose_path = os.path.join(save_folder, f"pose_{i}.txt")

        color_image = cv2.imread(color_img_path, cv2.IMREAD_COLOR)
        depth_image = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        cam_intr = load_intrinsics(intrinsics_path)
        cam_pose = np.loadtxt(cam_pose_path)

        # Open3D 的图像格式
        color_raw = o3d.geometry.Image(color_image)
        depth_raw = o3d.geometry.Image(depth_image)

        # 相机内参
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(width=color_image.shape[1], height=color_image.shape[0],
                                 fx=cam_intr[0, 0], fy=cam_intr[1, 1], cx=cam_intr[0, 2], cy=cam_intr[1, 2])

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, convert_rgb_to_intensity=False)

        volume.integrate(rgbd_image, intrinsic, np.linalg.inv(cam_pose))

        # 实时显示深度图和彩色图像
        cv2.imshow('Color Image', color_image)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=255 / depth_image.max()),
                                           cv2.COLORMAP_JET)
        cv2.imshow('Depth Image', depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # 将RGBDImage转换为点云
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

        # 可视化点云
        o3d.visualization.draw_geometries([pcd])
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(os.path.join(save_folder, "mesh.ply"), mesh)

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


if __name__ == "__main__":
    main()
