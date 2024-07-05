import time

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
import shutil


def clear_folder(folder_path):
    """清空指定文件夹中的所有内容"""
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def integrate_imu_data(accel_data, gyro_data, delta_t):
    # 初始化姿态矩阵（4x4齐次坐标矩阵）
    pose = np.eye(4)

    # 更新平移（简单积分加速度）
    pose[0, 3] += accel_data.x * delta_t ** 2
    pose[1, 3] += accel_data.y * delta_t ** 2
    pose[2, 3] += accel_data.z * delta_t ** 2

    # 更新旋转（简单积分陀螺仪数据）
    # 计算角速度的小角度近似
    wx, wy, wz = gyro_data.x * delta_t, gyro_data.y * delta_t, gyro_data.z * delta_t

    # 构建旋转矩阵的近似（使用罗德里格斯旋转公式的一阶近似）
    skew = np.array([[0, -wz, wy],
                     [wz, 0, -wx],
                     [-wy, wx, 0]])

    # 单位矩阵加上斜对称矩阵近似旋转矩阵
    rotation_approx = np.eye(3) + skew

    # 更新姿态矩阵的旋转部分
    pose[:3, :3] = np.dot(rotation_approx, pose[:3, :3])

    return pose


def save_pose(pose, filename):
    """将姿态矩阵保存到文件"""
    with open(filename, 'w') as file:
        for row in pose:
            file.write(' '.join(map(str, row)) + '\n')


def get_realsense_images():
    # 设置保存文件的文件夹
    save_folder = "captured_images"
    os.makedirs(save_folder, exist_ok=True)
    clear_folder(save_folder)

    # 创建realsense管道
    pipeline = rs.pipeline()
    config = rs.config()

    # 配置管道以流式传输彩色、深度流和IMU数据
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.accel)
    config.enable_stream(rs.stream.gyro)

    # 启动管道
    pipeline.start(config)
    Btime = time.time()

    try:
        frame_count = 0
        while True:
            # 等待一组帧：深度、颜色和IMU
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            accel_frame = frames.first(rs.stream.accel)
            gyro_frame = frames.first(rs.stream.gyro)

            if not depth_frame or not color_frame or not accel_frame or not gyro_frame:
                continue  # 如果缺少帧，则跳过当前循环

            # 读取IMU数据
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()

            # 将帧转换为numpy数组
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 深度图像的单位是毫米，转换为米
            depth_image_meters = depth_image * depth_frame.get_units()

            # 获取相机内参
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            cam_intr = np.array([
                [depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                [0, 0, 1]
            ])
            # 假设每帧的时间间隔约为1/30秒
            delta_t = 0.043
            cam_pose = integrate_imu_data(accel_data, gyro_data, delta_t)
            # 保存姿态
            pose_filename = os.path.join(save_folder, f"pose_{frame_count}.txt")
            save_pose(cam_pose, pose_filename)

            # 保存图像和内参到文件
            color_filename = os.path.join(save_folder, f"color_{frame_count}.png")
            depth_filename = os.path.join(save_folder, f"depth_{frame_count}.png")
            intrinsics_filename = os.path.join(save_folder, f"intrinsics_{frame_count}.json")

            cv2.imwrite(color_filename, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(depth_filename, depth_image_meters)  # 以可视化格式保存深度图像

            # 保存内参为JSON文件
            with open(intrinsics_filename, 'w') as f:
                json.dump(cam_intr.tolist(), f)

            frame_count += 1

            # 显示彩色和深度图像以及IMU数据
            cv2.imshow('RealSense Color Image', color_image)
            cv2.imshow('RealSense Depth Image',
                       cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET))
            print(
                f"Accel: {accel_data.x}, {accel_data.y}, {accel_data.z} | Gyro: {gyro_data.x}, {gyro_data.y}, {gyro_data.z}")

            # 按'q'键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 停止管道
        pipeline.stop()
        cv2.destroyAllWindows()
    print((time.time() - Btime) / frame_count)


if __name__ == "__main__":
    get_realsense_images()
