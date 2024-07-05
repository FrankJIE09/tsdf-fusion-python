# 导入Open3D库，Open3D是一个支持3D数据处理的开源库
import open3d as o3d

# 导入Numpy库，用于数值计算
import numpy as np

# 定义一个函数，用于准备点云数据
def prepare_data():
    # 定义点云数据文件路径
    pcd_data = {
        'paths': ["DemoICPPointClouds/cloud_bin_0.pcd", "DemoICPPointClouds/cloud_bin_1.pcd",
                  "DemoICPPointClouds/cloud_bin_2.pcd"]
    }
    # 读取第一个点云文件作为源点云
    source_raw = o3d.io.read_point_cloud(pcd_data['paths'][0])
    # 读取第二个点云文件作为目标点云
    target_raw = o3d.io.read_point_cloud(pcd_data['paths'][1])

    # 对源点云进行体素降采样，减少点的数量，提高处理速度
    source = source_raw.voxel_down_sample(voxel_size=0.02)
    # 对目标点云进行体素降采样
    target = target_raw.voxel_down_sample(voxel_size=0.02)

    # 定义一个初始变换矩阵，用于将源点云初步对齐到目标点云
    trans = [[0.862, 0.011, -0.507, 0.0], [-0.139, 0.967, -0.215, 0.7],
             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]]
    # 对源点云应用初始变换
    source.transform(trans)

    # 定义一个翻转变换矩阵，用于翻转点云
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    # 对源点云应用翻转变换
    source.transform(flip_transform)
    # 对目标点云应用翻转变换
    target.transform(flip_transform)

    # 返回处理后的源点云和目标点云
    return source, target

# 定义一个函数，用于演示非阻塞式的点云可视化
def demo_non_blocking_visualization():
    # 设置Open3D的详细输出级别为调试模式
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    # 准备点云数据
    source, target = prepare_data()

    # 创建一个可视化器对象
    vis = o3d.visualization.Visualizer()
    # 创建一个可视化窗口
    vis.create_window()

    # 添加源点云到可视化器
    vis.add_geometry(source)
    # 添加目标点云到可视化器
    vis.add_geometry(target)

    # 定义ICP算法的距离阈值
    threshold = 0.05
    # 定义ICP算法的迭代次数
    icp_iteration = 100
    # 定义是否保存每次迭代的屏幕截图
    save_image = False

    # 进行ICP迭代对齐
    for i in range(icp_iteration):
        # 使用点到平面的ICP算法进行注册
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source,  # 源点云数据，即需要对齐的点云
            target,  # 目标点云数据，即对齐的参考点云
            threshold,  # 距离阈值，表示ICP算法中对应点对的最大允许距离
            np.identity(4),  # 初始变换矩阵，用于将源点云初始对齐到目标点云，通常设置为单位矩阵
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            # 点到平面的变换估计方法，用于计算源点云和目标点云之间的刚性变换
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1)
            # ICP算法的收敛准则，在这里设置最大迭代次数为1，即每次只进行一次迭代
        )

        # 将计算得到的变换应用到源点云
        source.transform(reg_p2l.transformation)

        # 更新可视化器中的源点云
        vis.update_geometry(source)
        # 处理可视化器事件
        vis.poll_events()
        # 更新渲染器
        vis.update_renderer()

        # 如果需要保存屏幕截图，则保存当前迭代的截图
        if save_image:
            vis.capture_screen_image("temp_%04d.jpg" % i)

    # 销毁可视化窗口
    vis.destroy_window()

    # 将详细输出级别设置回信息模式
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)

# 如果当前脚本是主程序，则运行非阻塞可视化演示
if __name__ == '__main__':
    demo_non_blocking_visualization()
