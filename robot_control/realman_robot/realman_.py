from robotic_arm_package.robotic_arm import *
import sys

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:02:31 2024

@author: 86150
"""
from ikpy.chain import Chain
import numpy as np
from scipy.spatial.transform import Rotation

# 导入urdf模型
my_chain = Chain.from_urdf_file("../rm_65.urdf", active_links_mask=[False, True, True, True, True, True, True])


#   画八字
def demo1(robot):
    ret = robot.Movej_Cmd([18.44, -10.677, -124.158, -15, -71.455], 30)
    if ret != 0:
        print("设置初始位置失败:" + str(ret))
        sys.exit()

    for num in range(0, 3):
        po1 = [0.186350, 0.062099, 0.2, 3.141, 0, 1.569]

        ret = robot.Movel_Cmd(po1, 30)
        if ret != 0:
            print("Movel_Cmd 1 失败:" + str(ret))
            sys.exit()

        po2 = [0.21674, 0.0925, 0.2, 3.141, 0, 1.569]
        po3 = [0.20785, 0.114, 0.2, 3.141, 0, 1.569]

        ret = robot.Movec_Cmd(po2, po3, 30)
        if ret != 0:
            print("Movec_Cmd 1 失败:" + str(ret))
            sys.exit()

        po4 = [0.164850, 0.157, 0.2, 3.141, 0, 1.569]

        ret = robot.Movel_Cmd(po4, 30)
        if ret != 0:
            print("Movel_Cmd 2 失败:" + str(ret))
            sys.exit()

        po5 = [0.186350, 0.208889, 0.2, 3.141, 0, 1.569]
        po6 = [0.20785, 0.157, 0.2, 3.141, 0, 1.569]

        ret = robot.Movec_Cmd(po5, po6, 30)
        if ret != 0:
            print("Movel_Cmd 2 失败:" + str(ret))
            sys.exit()

    print("cycle draw 8 demo success")


#   获取机械臂状态、坐标系
def demo2(robot):
    # 获取机械臂当前状态
    ret, joint, pose, arm_err, sys_err = robot.Get_Current_Arm_State(retry=1)
    if ret != 0:
        print("获取机械臂当前状态失败：" + str(ret))
        sys.exit()

    print("当前关节角度：" + str(joint))
    print("错误码: " + str(arm_err) + str(sys_err))

    # 获取当前坐标系
    retval, frame = robot.Get_Current_Work_Frame(retry=1)
    if retval == 0:
        print('当前工作坐标系：', frame.frame_name.name)
        print('当前工作坐标系位置：', frame.pose.position.x, frame.pose.position.y, frame.pose.position.z)
    else:
        print(f'获取当前坐标系失败:{retval}')
        sys.exit()

    #  设置工作坐标系
    retval = robot.Manual_Set_Work_Frame('new', [0.1, 0.2, 0.3, 3.0, 0.2, 0.1])
    if retval != 0:
        print(f'设置坐标系失败:{retval}')
        # sys.exit()

    # 切换当前工作坐标系
    robot.Change_Work_Frame('new')
    # 获取当前工作坐标系
    retval, frame = robot.Get_Current_Work_Frame(retry=1)
    if retval == 0:
        print('当前工作坐标系：', frame.frame_name.name)
        print('当前工作坐标系位置：', frame.pose.position.x, frame.pose.position.y, frame.pose.position.z)
    else:
        print(f'获取当前坐标系失败:{retval}')
        sys.exit()

    # 获取指定坐标系
    retval, frame = robot.Get_Given_Work_Frame('new', retry=1)
    if retval == 0:
        print('指定工作坐标系位置：', frame)
    else:
        print(f'获取指定坐标系失败:{retval}')
        sys.exit()


# 夹爪使用例程（需机械臂末端安装夹爪）
def demo3(robot):
    #   回零位
    zero = [0, 0, 0, 0, 0, 0]
    ret = robot.Movej_Cmd(zero, 20)
    if ret != 0:
        print("回零位失败：" + str(ret))
        sys.exit()

    #   张开夹爪，抓取位置
    robot.Set_Gripper_Release(500)
    if ret != 0:
        print("张开夹爪失败：" + str(ret))
        sys.exit()

    joint1 = [4.61, 93.549, 75.276, -10.098, -76.508, 57.224]
    ret = robot.Movej_Cmd(joint1, 20)
    if ret != 0:
        print("到达抓取位置失败：" + str(ret))
        sys.exit()

    #   抓取
    ret = robot.Set_Gripper_Pick_On(500, 500)
    if ret != 0:
        print("抓取失败：" + str(ret))
        sys.exit()

    #   放置
    joint2 = [-106.244, 67.172, 96.15, -10.385, -71.097, 58.243]

    ret = robot.Movej_Cmd(joint2, 20)
    if ret != 0:
        print("到达放置位置失败：" + str(ret))
        sys.exit()

    robot.Set_Gripper_Release(200)
    if ret != 0:
        print("放置失败：" + str(ret))
        sys.exit()

    #   回零位
    ret = robot.Movej_Cmd(zero, 20, 0, 1)
    if ret != 0:
        print("回零位失败：" + str(ret))
        sys.exit()
    print("夹爪抓取成功")


# 位置示教例程
def demo4(robot):
    # 初始位置
    joint = [0, -20, -70, 0, -90, 0]
    zero = [0, 0, 0, 0, 0, 0]

    robot.Movej_Cmd(joint, 20)

    # 切换示教坐标系为基坐标系
    robot.Set_Teach_Frame(0)
    #   位置示教
    ret = robot.Pos_Teach_Cmd(2, 1, 10)
    time.sleep(2)
    if ret != 0:
        print("Z轴正方向示教失败：" + str(ret))
        sys.exit()

    ret = robot.Teach_Stop_Cmd()
    if ret != 0:
        print("停止示教失败：" + str(ret))
        sys.exit()

    # 切换示教坐标系为工具坐标系
    robot.Set_Teach_Frame(1)

    #   位置示教
    ret = robot.Pos_Teach_Cmd(2, 1, 20)
    time.sleep(2)
    if ret != 0:
        print("Z轴正方向示教失败：" + str(ret))
        sys.exit()

    ret = robot.Teach_Stop_Cmd()
    if ret != 0:
        print("停止示教失败：" + str(ret))
        sys.exit()

    print("Z轴位置示教成功")


# 透传
def demo5(robot):
    # 读取文件内容，文件中的点位为拖动示教所得
    with open('output-65.txt', 'r') as f:
        lines = f.readlines()

    # 转换为浮点数列表
    point = []
    for line in lines:
        nums = line.strip().split(',')
        point.append([float(num) for num in nums])

    # 运动到透传的第一个点位
    num_lines = len(point)
    robot.Movej_Cmd(point[0], 20, 0, True)

    # 低跟随透传

    for i in range(num_lines):
        robot.Movej_CANFD(point[i], False)
        time.sleep(0.005)

    print("透传结束")


# 力位混合控制透传（需为六维力版本机械臂）
def demo6(robot):
    # 读取文件内容
    with open('output-65.txt', 'r') as f:
        lines = f.readlines()

    # 转换为浮点数列表
    zero = []
    for line in lines:
        nums = line.strip().split(',')
        zero.append([float(num) for num in nums])

    # 使用文件行数
    num_lines = len(zero)
    robot.Movej_Cmd(zero[0], 20, 0, True)

    # 开启透传力位混合控制补偿模式
    robot.Start_Force_Position_Move()
    time.sleep(1)

    for i in range(num_lines):
        robot.Force_Position_Move_Joint(zero[i], 1, 0, 2, 2, False)
        print(zero[i])
        time.sleep(0.001)

    robot.Stop_Force_Position_Move()


# 正逆解示例
def demo7(robot):
    #   初始位置
    ret, joint, pose, arm_err, sys_err = robot.Get_Current_Arm_State(retry=1)
    joint_with_zero = [0] + joint
    joint_in_radians = np.deg2rad(joint_with_zero)
    T = my_chain.forward_kinematics(joint_in_radians)
    print("T:", T)
    target_matrix = T
    #   逆解
    target_matrix[0, -1] = target_matrix[0, -1]
    target_matrix[:3,:3] = Rotation.from_euler('xyz', np.array([np.pi / 2, 0, np.pi / 2])).as_matrix()@Rotation.from_euler('xyz', np.array([0, 0, np.pi /6])).as_matrix()
    ik_joint = my_chain.inverse_kinematics_frame(target_matrix,
                                                 initial_position=np.deg2rad([0] + joint), orientation_mode='all')
    print(ik_joint)
    ret = robot.Movej_Cmd(np.rad2deg(ik_joint[1:]), 2, r=0, block=True)
    print("Movej_Cmd ret:" + str(ret))


if __name__ == "__main__":
    def mcallback(data):
        print("MCallback MCallback MCallback")
        # 判断接口类型
        if data.codeKey == MOVEJ_CANFD_CB:  # 角度透传
            print("透传结果:", data.errCode)
            print("当前角度:", data.joint[0], data.joint[1], data.joint[2], data.joint[3], data.joint[4], data.joint[5])
        elif data.codeKey == MOVEP_CANFD_CB:  # 位姿透传
            print("透传结果:", data.errCode)
            print("当前角度:", data.joint[0], data.joint[1], data.joint[2], data.joint[3], data.joint[4], data.joint[5])
            print("当前位姿:", data.pose.position.x, data.pose.position.y, data.pose.position.z, data.pose.euler.rx,
                  data.pose.euler.ry, data.pose.euler.rz)
        elif data.codeKey == FORCE_POSITION_MOVE_CB:  # 力位混合透传
            print("透传结果:", data.errCode)
            print("当前力度：", data.nforce)


    # 连接机械臂，注册回调函数
    callback = CANFD_Callback(mcallback)
    robot = Arm(RM65, "192.168.1.18", callback)
    robot.Set_Collision_Stage(6)
    # API版本信息
    print(robot.API_Version())

    # 运行示例
    demo7(robot)
    demo2(robot)

    # 断开连接
    robot.RM_API_UnInit()
    robot.Arm_Socket_Close()
