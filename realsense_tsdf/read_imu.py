import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


# 姿态计算函数
def compute_orientation(accel, gyro, delta_t):
    """
    计算姿态，这里使用简单的积分来模拟姿态计算。
    实际应用应使用更复杂的算法，如卡尔曼滤波器。
    """
    global orientation
    # 陀螺仪积分（简单示例，未考虑误差累积）
    orientation += gyro * delta_t
    return orientation


# 初始化全局变量
orientation = np.array([0.0, 0.0, 0.0])  # 使用浮点数初始化
orientation = orientation.astype(np.float64)  # 转换为float64类型

# 设置绘图
fig, ax = plt.subplots()
xdata, ydata, zdata = [], [], []
ln, = plt.plot([], [], 'ro', animated=True)


def init():
    ax.set_xlim(0, 10)
    ax.set_ylim(-180, 180)
    return ln,


def update(frame):
    global orientation
    delta_t = 0.1  # 假设每帧间隔0.1秒

    # 模拟读取IMU数据
    gyro = np.random.normal(0, 0.1, 3)  # 生成随机陀螺仪数据
    accel = np.random.normal(0, 0.1, 3)  # 生成随机加速度数据

    # 计算当前姿态
    new_orientation = compute_orientation(accel, gyro, delta_t)

    # 更新绘图数据
    xdata.append(frame * delta_t)
    ydata.append(new_orientation[0])
    zdata.append(new_orientation[1])
    ln.set_data(xdata, ydata)
    ax.set_xlim(0, frame * delta_t + 10)
    return ln,


ani = FuncAnimation(fig, update, frames=np.arange(100), init_func=init, blit=True)
plt.show()
