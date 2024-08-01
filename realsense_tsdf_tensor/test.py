import pyrealsense2 as rs
import time
import numpy as np
import matplotlib.pyplot as plt

# 初始化RealSense管道
pipeline = rs.pipeline()
config = rs.config()

# 启用加速度计和陀螺仪流
config.enable_stream(rs.stream.accel)
config.enable_stream(rs.stream.gyro)

# 开始流
pipeline.start(config)

# 初始位置和速度
position = np.array([0.0, 0.0, 0.0])
velocity = np.array([0.0, 0.0, 0.0])

# 时间步长
dt = 1

# 重力加速度
gravity = np.array([0.0, -9.81, 0])

# 数据存储
positions = []
rolls = []
pitches = []
yaws = []


def get_rotation_matrix(gyro_data, dt):
    """根据陀螺仪数据计算旋转矩阵"""
    angle = np.array([gyro_data.x, gyro_data.y, gyro_data.z]) * dt
    angle_mag = np.linalg.norm(angle)
    if angle_mag > 0:
        axis = angle / angle_mag
        cos_angle = np.cos(angle_mag)
        sin_angle = np.sin(angle_mag)
        one_minus_cos = 1.0 - cos_angle
        axis_skew = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        rotation_matrix = (
                cos_angle * np.eye(3) +
                sin_angle * axis_skew +
                one_minus_cos * np.outer(axis, axis)
        )
        return rotation_matrix
    else:
        return np.eye(3)


# 初始旋转矩阵
rotation_matrix = np.eye(3)
last = time.time()
loop = False
try:
    while True:
        # 等待一帧
        now = time.time()
        dt = now - last
        last = now
        frames = pipeline.wait_for_frames()

        # 获取加速度计数据
        accel_frame = frames.first_or_default(rs.stream.accel)
        if accel_frame:
            accel_data = accel_frame.as_motion_frame().get_motion_data()

        # 获取陀螺仪数据
        gyro_frame = frames.first_or_default(rs.stream.gyro)
        if not loop:
            accel_init = np.array([accel_data.x, accel_data.y, accel_data.z])
            gyro_data_init = gyro_frame.as_motion_frame().get_motion_data()
            loop = True
        if gyro_frame:
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            gyro_data.x -= gyro_data_init.x
            gyro_data.y -= gyro_data_init.y
            gyro_data.z -= gyro_data_init.z

            # 使用陀螺仪数据更新旋转矩阵
            rotation_matrix = get_rotation_matrix(gyro_data, dt) @ rotation_matrix

        # 将加速度转换到全球坐标系并减去重力影响
        if accel_frame:
            accel = np.array([accel_data.x, accel_data.y, accel_data.z])
            accel = accel-accel_init
            global_accel = rotation_matrix @ accel

            # 积分加速度以得到速度
            velocity += global_accel * dt

            # 积分速度以得到位移
            position += velocity * dt

            # 打印估算的位移
            print(f"Estimated Position: x={position[0]}, y={position[1]}, z={position[2]}")

            # 存储位置数据
            positions.append(position.copy())

            # 计算姿态
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2))
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            rolls.append(np.degrees(roll))
            pitches.append(np.degrees(pitch))
            yaws.append(np.degrees(yaw))

        # 暂停以保持固定的采样频率
        # time.sleep(1)

finally:
    # 停止流
    pipeline.stop()

    # 转换位置数据为numpy数组
    positions = np.array(positions)

    # 绘制位置数据
    plt.figure()
    plt.plot(positions[:, 0], label='X Position')
    plt.plot(positions[:, 1], label='Y Position')
    plt.plot(positions[:, 2], label='Z Position')
    plt.xlabel('Time')
    plt.ylabel('Position (meters)')
    plt.legend()
    plt.title('Position Over Time')

    # 绘制姿态数据
    plt.figure()
    plt.plot(rolls, label='Roll')
    plt.plot(pitches, label='Pitch')
    plt.plot(yaws, label='Yaw')
    plt.xlabel('Time')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.title('Orientation Over Time')

    # 显示图表
    plt.show()
