import open3d as o3d
import os
import time
from tqdm import tqdm


def realtime_visualize_pcds(folder_path):

    pcd_files = sorted([file for file in os.listdir(folder_path) if file.endswith('.pcd')])
    vis = o3d.visualization.Visualizer()
    vis.create_window("Realtime PCD Viewer", width=800, height=600)
    first = True
    # 用于跟踪当前显示的几何形状
    displayed_geometry = None

    for pcd_file in tqdm(pcd_files, desc="Visualizing PCDs"):
        pcd_path = os.path.join(folder_path, pcd_file)
        pcd = o3d.io.read_point_cloud(pcd_path)
        if pcd.is_empty():
            print(f"Failed to load {pcd_file}.")
            continue  # 如果文件加载失败，跳过并继续下一个文件

        # 移除上一个显示的点云
        if displayed_geometry is not None:
            vis.remove_geometry(displayed_geometry, reset_bounding_box=False)

        # 添加新的点云并重设视点
        vis.add_geometry(pcd)
        vis.reset_view_point(True)
        displayed_geometry = pcd
        # 设置视角
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])  # 设置前视角
        ctr.set_lookat([0, 0, 0])  # 设置观察目标点
        ctr.set_up([0, -1, 0])     # 设置上方向
        ctr.set_zoom(0.8)          # 设置缩放比例

        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)  # 控制每帧展示的时间

    # 阻塞代码，保持窗口打开
    print("Press 'q' to close the window...")
    vis.run()  # 让视窗保持运行，直到用户关闭它
    # vis.destroy_window()


if __name__ == "__main__":
    realtime_visualize_pcds("pcd_folder")
