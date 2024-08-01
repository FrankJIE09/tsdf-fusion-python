import os
import numpy as np
import open3d as o3d
from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import logging as log
class PipelineModel:
    """处理从RGBD摄像头或文件读取数据，将RGBD数据转换为点云，并进行可视化。"""

    def __init__(self, device='cpu:0'):
        """初始化管线，配置设备和内外参数。"""
        self.device = 'cuda:0' if o3d.core.cuda.is_available() and device == 'cuda:0' else 'cpu:0'
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='Worker')
        self.setup_camera_intrinsics()
        self.tsdf_volume = self.create_tsdf_volume()

    def setup_camera_intrinsics(self):
        """设置摄像头内参。"""
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, 525.0, 525.0, 319.5, 239.5)

    def create_tsdf_volume(self):
        """创建TSDF体积用于3D重建。"""
        return o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.002,
            sdf_trunc=0.06,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    def process_frame(self, frame_id, rgbd_video_path):
        """处理单个帧，转换为点云并更新TSDF体积。"""
        color_image = o3d.io.read_image(os.path.join(rgbd_video_path, f"{frame_id}_color.jpg"))
        depth_image = o3d.io.read_image(os.path.join(rgbd_video_path, f"{frame_id}_depth.png"))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.intrinsics)
        self.tsdf_volume.integrate(rgbd_image, self.intrinsics, np.eye(4))
        return pcd

    def run(self, rgbd_video_path):
        """执行处理循环，处理视频中的所有帧并提取TSDF体积生成的网格。"""
        frame_ids = [filename.split('_')[0] for filename in os.listdir(rgbd_video_path) if 'color' in filename]
        for frame_id in sorted(set(frame_ids)):
            self.process_frame(frame_id, rgbd_video_path)
        mesh = self.tsdf_volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])
def main():
    """主程序入口，处理命令行参数并启动处理管线。"""
    log.basicConfig(level=log.INFO)
    parser = ArgumentParser(description='3D Depth Video Processing Pipeline with TSDF Integration',
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='cuda:0', help='设备选择，支持cpu:0或cuda:0')
    parser.add_argument('--video-path', required=False, help='RGBD视频文件路径')
    args = parser.parse_args()

    pipeline = PipelineModel(device=args.device)
    pipeline.run(args.video_path)

if __name__ == "__main__":
    main()
