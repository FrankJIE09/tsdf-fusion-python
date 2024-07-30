from robotic_arm_package.robotic_arm import *
import sys
from ikpy.chain import Chain
import numpy as np
from scipy.spatial.transform import Rotation


class RoboticArmRecalibration:
    def __init__(self, urdf_file="../rm_65.urdf", robot_ip="192.168.1.18", ):
        self.my_chain = Chain.from_urdf_file(urdf_file, active_links_mask=[False, True, True, True, True, True, True])
        self.robot_ip = robot_ip
        self.callback_function = mcallback
        self.robot = None
        self.connect_robot()
        self.print_api_version()

    def connect_robot(self):
        callback = CANFD_Callback(self.callback_function)
        self.robot = Arm(RM65, self.robot_ip, callback)
        self.robot.Set_Collision_Stage(6)
        self.print_api_version()

    def print_api_version(self):
        print(self.robot.API_Version())

    def get_current_joint(self):
        ret, joint, pose, arm_err, sys_err = self.robot.Get_Current_Arm_State(retry=1)
        return joint

    def perform_forward_kinematics(self, joint):
        joint_with_zero = [0] + joint
        joint_in_radians = np.deg2rad(joint_with_zero)
        T = self.my_chain.forward_kinematics(joint_in_radians)
        return T

    def move_robot(self, target_matrix):
        joint = np.deg2rad(self.robot.get_current_joint())
        target_matrix[0, -1] = target_matrix[0, -1]
        target_matrix[:3, :3] = Rotation.from_euler('xyz',
                                                    np.array([np.pi / 2, 0,
                                                              np.pi / 2])).as_matrix() @ Rotation.from_euler(
            'xyz', np.array([0, 0, np.pi / 6])).as_matrix()
        ik_joint = self.my_chain.inverse_kinematics_frame(target_matrix,
                                                          initial_position=np.deg2rad([0] + joint),
                                                          orientation_mode='all')
        self.robot.Movej_Cmd(np.rad2deg(ik_joint[1:]), 2, r=0, block=True)

    def disconnect_robot(self):
        self.robot.RM_API_UnInit()
        self.robot.Arm_Socket_Close()


def mcallback(data):
    pass
