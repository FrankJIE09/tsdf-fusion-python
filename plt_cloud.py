import open3d as o3d

# 读取点云数据
pcd = o3d.io.read_point_cloud("pc.ply")

# 设置窗口属性并可视化点云
o3d.visualization.draw_geometries(
    [pcd],                              # 要显示的几何对象列表
    window_name="Point Cloud",          # 窗口名称
    width=1920,                          # 窗口宽度
    height=1080,                         # 窗口高度
    left=50,                            # 窗口左上角横坐标
    top=50,                             # 窗口左上角纵坐标
    point_show_normal=False,            # 是否显示点的法线
    mesh_show_wireframe=False,          # 是否显示网格线框
    mesh_show_back_face=False           # 是否显示背面的网格
)
