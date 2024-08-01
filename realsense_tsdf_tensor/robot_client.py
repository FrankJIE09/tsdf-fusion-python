import numpy as np
import websockets
import cbor2
import json
import time
import asyncio


def get_geomEB_value(data_list):
    """
    从包含多个字典的列表中提取 'geomEB' 类型的 'value'。

    参数:
    data_list (list): 包含多个字典的列表，每个字典至少有 'type' 和 'value' 两个键。

    返回:
    list or None: 返回找到的 'value' 列表，如果没有找到 'geomEB' 类型，返回 None。
    """
    for item in data_list:
        # 确保 item 是字典并且包含 'type' 和 'value' 键
        if isinstance(item, dict) and 'type' in item and 'value' in item:
            if item['type'] == 'geomEB':
                return item['value']
        else:
            print(f"Unexpected item format or type: {item}")
    return None


def euclidean_distance(point1, point2):
    """
    计算两个点之间的欧氏距离。
    point1 和 point2 都是包含三个坐标 (x, y, z) 的列表或 numpy 数组。
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


class RobotClient:
    def __init__(self, uri):
        """
        初始化 RobotClient 类，设置 WebSocket 连接的 URI。
        """
        self.uri = uri

    async def connect(self):
        """
        建立 WebSocket 连接。
        """
        self.websocket = await websockets.connect(self.uri)

    async def send(self, json_data):
        """
        将 JSON 数据编码为 CBOR 格式并通过 WebSocket 发送。
        """
        try:
            # 将数据编码为 CBOR 格式
            cbor_data = cbor2.dumps(json_data)
            # print(f"Sending JSON data: {json.dumps(json_data)}")

            # 设置为二进制模式发送数据
            await self.websocket.send(cbor_data)

            # 同步读取响应
            response = await self.websocket.recv()
            return self.wait_for_send(response)
        except Exception as e:
            print(f"Error sending data: {e}")
            return None

    def wait_for_send(self, response):
        """
        解码接收到的 CBOR 响应数据并打印。
        """
        try:
            decoded_response = cbor2.loads(response)
            # print(f"Received response: {decoded_response}")
            return decoded_response
        except Exception as e:
            print(f"Error decoding response: {e}")
            return None

    async def close(self):
        """
        关闭 WebSocket 连接。
        """
        await self.websocket.close()

    async def set_max_scale(self, vel_scale, n=1190):
        """
        设置速度/加速度系数

        参数:
        vel_scale (int): 速度值（%）
        n (int): 命令编号，默认为 1190。
        """
        json_data = {
            "c": "robot",
            "a": "setMaxScale",
            "d": [1, vel_scale, 0],
            "n": n
        }
        return await self.send(json_data)

    async def coord_sys(self, coord_type, coord=None, key=None, value=None, use_name=None, equip=0, name=None, n=1193):
        """
        坐标系管理

        参数:
        coord_type (int): 类型（1=获取所有坐标信息，2=修改/增加，3=使用，4=删除）
        coord (str): 坐标类型（work 或 tool）
        key (str): 坐标系名称
        value (list): 坐标系值
        use_name (str): 使用的坐标系名称
        equip (int): 设备编号
        name (str): 要删除的坐标系名称
        n (int): 命令编号，默认为 1193。
        """
        json_data = {
            "c": "robot",
            "a": "coordSys",
            "d": {"type": coord_type},
            "n": n
        }
        if coord_type == 2:
            json_data["d"].update({"coord": coord, "key": key, "value": value})
        elif coord_type == 3:
            json_data["d"].update({"coord": coord, "useName": use_name, "equip": equip})
        elif coord_type == 4:
            json_data["d"].update({"coord": coord, "name": name})

        return await self.send(json_data)

    async def set_zero_pos(self, zero_positions, n=1146):
        """
        设置零位

        参数:
        zero_positions (list): 零位坐标 [zeroJ1, zeroJ2, zeroJ3, zeroJ4, zeroJ5, zeroJ6]
        n (int): 命令编号，默认为 1146。
        """
        json_data = {
            "c": "robot",
            "a": "setZeroPos",
            "d": {
                "equip": 0,
                "dZeroQ": zero_positions
            },
            "n": n
        }
        return await self.send(json_data)

    async def set_robot_param(self, dh_params, n=1261):
        """
        设置DH参数

        参数:
        dh_params (list): DH参数数组（8个）
        n (int): 命令编号，默认为 1261。
        """
        json_data = {
            "c": "robot",
            "a": "setRobotParam",
            "d": {
                "equip": 0,
                "DH": dh_params
            },
            "n": n
        }
        return await self.send(json_data)

    async def set_load_dyn(self, mass, center, inertia=[0, 0, 0, 0, 0, 0], n=1160):
        """
        设置TCP负载

        参数:
        mass (float): 质量
        center (list): 质心位置 [X, Y, Z]
        inertia (list): 惯性 [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
        n (int): 命令编号，默认为 1160。
        """
        json_data = {
            "c": "robot",
            "a": "setLoadDyn",
            "d": {
                "equip": 0,
                "mass": mass,
                "gCenter": center,
                "gInertia": inertia
            },
            "n": n
        }
        return await self.send(json_data)

    async def set_gravity_acc(self, direction, n=1161):
        """
        设置重力加速度(相对基座坐标系)

        参数:
        direction (list): 重力加速度方向 [X, Y, Z]
        n (int): 命令编号，默认为 1161。
        """
        json_data = {
            "c": "robot",
            "a": "setGravityAcc",
            "d": {
                "equip": 0,
                "dir": direction
            },
            "n": n
        }
        return await self.send(json_data)

    async def set_collision_factor(self, factor, n=1147):
        """
        安全等级设置

        参数:
        factor (int): 安全等级系数，范围 1-50
        n (int): 命令编号，默认为 1147。
        """
        json_data = {
            "c": "robot",
            "a": "setCollisionFactor",
            "d": factor * 10,
            "n": n
        }
        return await self.send(json_data)

    async def get_info(self, n=1227):
        """
        获取机械臂所有状态信息

        参数:
        n (int): 命令编号，默认为 1227。
        """
        json_data = {
            "c": "robot",
            "a": "getInfo",
            "d": 0,
            "n": n
        }
        return await self.send(json_data)

    async def get_fault_code(self, n=1254):
        """
        获取机器人运行错误码

        参数:
        n (int): 命令编号，默认为 1254。
        """
        json_data = {
            "c": "robot",
            "a": "getFaultCode",
            "d": True,
            "n": n
        }
        return await self.send(json_data)

    async def get_geom(self, n=1104):
        """
        获取机器人设置点位信息

        参数:
        n (int): 命令编号，默认为 1104。
        """
        json_data = {
            "c": "robot",
            "a": "getGeom",
            "d": 0,
            "n": n
        }
        return await self.send(json_data)

    async def get_zero_pos(self, n=1029):
        """
        获取关节角度(设置零位时使用)

        参数:
        n (int): 命令编号，默认为 1029。
        """
        json_data = {
            "c": "robot",
            "a": "getZeroPos",
            "d": 0,
            "n": n
        }
        return await self.send(json_data)

    async def get_load_dyn(self, n=1071):
        """
        获取服务器的TCP负载值

        参数:
        n (int): 命令编号，默认为 1071。
        """
        json_data = {
            "c": "robot",
            "a": "getLoadDyn",
            "d": {"equip": 0},
            "n": n
        }
        return await self.send(json_data)

    async def get_joint_abs_pos(self, n=1264):
        """
        获取关节绝对零位

        参数:
        n (int): 命令编号，默认为 1264。
        """
        json_data = {
            "c": "robot",
            "a": "getJointAbsPos",
            "d": {"equip": 0},
            "n": n
        }
        return await self.send(json_data)

    async def get_versions(self, n=10006):
        """
        获取版本号

        参数:
        n (int): 命令编号，默认为 10006。
        """
        json_data = {
            "c": "robot",
            "a": "getVersions",
            "n": n
        }
        return await self.send(json_data)

    async def get_geom_end_in_base(self, n=1010):
        """
        获取实际末端执行器相对于基座的位姿

        参数:
        n (int): 命令编号，默认为 1010。
        """
        json_data = {
            "c": "robot",
            "a": "getGeomEndInBase",
            "d": {"equip": 0},
            "n": n
        }
        return await self.send(json_data)

    async def get_desire_geom(self, n=1103):
        """
        获取期望位姿

        参数:
        n (int): 命令编号，默认为 1103。
        """
        json_data = {
            "c": "robot",
            "a": "getDesireGeom",
            "d": 0,
            "n": n
        }
        return await self.send(json_data)

    async def power(self, state, n=1147):
        """
        机器人上电/下电

        参数:
        state (bool): 机器人上电(True)/下电(False)
        n (int): 命令编号，默认为 1147。
        """
        json_data = {
            "c": "robot",
            "a": "power",
            "d": state,
            "n": n
        }
        return await self.send(json_data)

    async def start(self, state, n=1199):
        """
        机器人启动/停止

        参数:
        state (bool): 机器人启动(True)/停止(False)
        n (int): 命令编号，默认为 1199。
        """
        json_data = {
            "c": "robot",
            "a": "start",
            "d": state,
            "n": n
        }
        return await self.send(json_data)

    async def clear_fault(self, n=1137):
        """
        机器人错误清除

        参数:
        n (int): 命令编号，默认为 1137。
        """
        json_data = {
            "c": "robot",
            "a": "clearFault",
            "d": True,
            "n": n
        }
        return await self.send(json_data)

    async def switch_to_free(self, state, n=1151):
        """
        自由拖动

        参数:
        state (bool): 开启自由拖动(True)/关闭自由拖动(False)
        n (int): 命令编号，默认为 1151。
        """
        json_data = {
            "c": "robot",
            "a": "switchToFree",
            "d": state,
            "n": n
        }
        return await self.send(json_data)

    async def switch_to_reverse(self, n=1163):
        """
        电机方向驱动

        参数:
        n (int): 命令编号，默认为 1163。
        """
        json_data = {
            "c": "robot",
            "a": "switchToReverse",
            "d": True,
            "n": n
        }
        return await self.send(json_data)

    async def switch_to_close_robot(self, n=1168):
        """
        关闭机器人

        参数:
        n (int): 命令编号，默认为 1168。
        """
        json_data = {
            "c": "robot",
            "a": "switchToCloseRobot",
            "d": True,
            "n": n
        }
        return await self.send(json_data)

    async def had_save_data_from_hmi(self, state, n=1012):
        """
        HMI是否完成数据保存

        参数:
        state (bool): 已保存(True)/未保存(False)
        n (int): 命令编号，默认为 1012。
        """
        json_data = {
            "c": "robot",
            "a": "hadSaveDataFromHMI",
            "d": state,
            "n": n
        }
        return await self.send(json_data)

    async def abort_move(self, n=1102):
        """
        机器人停止运动

        参数:
        n (int): 命令编号，默认为 1102。
        """
        json_data = {
            "c": "robot",
            "a": "abortMove",
            "d": 0,
            "n": n
        }
        return await self.send(json_data)

    async def reset_angles(self, angles, n=1136):
        """
        机器人归零状态

        参数:
        angles (list): 关节角度值
        n (int): 命令编号，默认为 1136。
        """
        json_data = {
            "c": "robot",
            "a": "resetAngles",
            "d": angles,
            "n": n
        }
        return await self.send(json_data)

    async def move_to(self, target, speed, n=1172):
        """
        机器人移动到指定位置

        参数:
        target (list): 目标位置
        speed (int): 移动速度
        n (int): 命令编号，默认为 1172。
        """
        json_data = {
            "c": "robot",
            "a": "moveTo",
            "d": {"target": target, "speed": speed},
            "n": n
        }
        return await self.send(json_data)

    async def start_joint_rot(self, joint, speed, n=1182):
        """
        机器人指定关节移动

        参数:
        joint (int): 关节编号
        speed (int): 移动速度
        n (int): 命令编号，默认为 1182。
        """
        json_data = {
            "c": "robot",
            "a": "startJointRot",
            "d": {"joint": joint, "speed": speed},
            "n": n
        }
        return await self.send(json_data)

    async def start_cart_trans(self, direction, speed, n=1184):
        """
        机器人平移运动

        参数:
        direction (list): 平移方向
        speed (int): 移动速度
        n (int): 命令编号，默认为 1184。
        """
        json_data = {
            "c": "robot",
            "a": "startCartTrans",
            "d": {"direction": direction, "speed": speed},
            "n": n
        }
        return await self.send(json_data)

    async def move_line(self, geom, equip=0, coord_name=["_Base", "_Flange"], move_info=[1, 1, 1, 1], n=1128):
        """
        发送直线移动的 JSON 数据。

        参数:
        geom (list): 直线移动几何信息:x,y,z,r,p,y
        equip (int): 设备编号，默认为 0。
        coord_name (list): 坐标名称，默认为 ["_Base", "_Flange"]。
        move_info (list): 移动信息，默认为 [1, 1, 1, 1]。
        n (int): 命令编号，默认为 1128。
        """
        if isinstance(geom, np.ndarray):
            # 如果是numpy数组，则创建一个新数组，在前面加上六个0
            geom = geom.tolist()
        geom = geom + [0, 0, 0, 0, 0, 0]
        json_data = {
            "c": "robot",
            "a": "moveLine",
            "d": {
                "equip": equip,
                "geom": geom,
                "coordName": coord_name,
                "moveInfo": move_info
            },
            "n": n
        }
        return await self.send(json_data)

    async def start_cart_rot(self, direction, speed, n=1195):
        """
        机器人旋转运动

        参数:
        direction (list): 旋转方向
        speed (int): 旋转速度
        n (int): 命令编号，默认为 1195。
        """
        json_data = {
            "c": "robot",
            "a": "startCartRot",
            "d": {"direction": direction, "speed": speed},
            "n": n
        }
        return await self.send(json_data)

    async def move_arc(self, point1, point2, speed, n=1194):
        """
        机器人做圆弧运动

        参数:
        point1 (list): 圆弧起点
        point2 (list): 圆弧终点
        speed (int): 运动速度
        n (int): 命令编号，默认为 1194。
        """
        json_data = {
            "c": "robot",
            "a": "moveArc",
            "d": {"point1": point1, "point2": point2, "speed": speed},
            "n": n
        }
        return await self.send(json_data)

    async def pause_move(self, n=1200):
        """
        暂停机械臂运动

        参数:
        n (int): 命令编号，默认为 1200。
        """
        json_data = {
            "c": "robot",
            "a": "pauseMove",
            "d": 0,
            "n": n
        }
        return await self.send(json_data)

    async def resume_move(self, n=1201):
        """
        恢复机械臂运动

        参数:
        n (int): 命令编号，默认为 1201。
        """
        json_data = {
            "c": "robot",
            "a": "resumeMove",
            "d": 0,
            "n": n
        }
        return await self.send(json_data)

    async def is_move_end(self, n=1202):
        """
        判断运动是否停止

        参数:
        n (int): 命令编号，默认为 1202。
        """
        json_data = {
            "c": "robot",
            "a": "isMoveEnd",
            "d": 0,
            "n": n
        }
        res = await self.send(json_data)
        print(res)
        return res

    async def start_joint_feed_rot(self, joint, speed, n=1203):
        """
        关节进给运动

        参数:
        joint (int): 关节编号
        speed (int): 移动速度
        n (int): 命令编号，默认为 1203。
        """
        json_data = {
            "c": "robot",
            "a": "startJointFeedRot",
            "d": {"joint": joint, "speed": speed},
            "n": n
        }
        return await self.send(json_data)

    async def start_cart_feed_trans(self, direction, speed, n=1204):
        """
        平移进给

        参数:
        direction (list): 平移方向
        speed (int): 移动速度
        n (int): 命令编号，默认为 1204。
        """
        json_data = {
            "c": "robot",
            "a": "startCartFeedTrans",
            "d": {"direction": direction, "speed": speed},
            "n": n
        }
        return await self.send(json_data)

    async def start_cart_feed_rot(self, direction, speed, n=1205):
        """
        旋转进给

        参数:
        direction (list): 旋转方向
        speed (int): 旋转速度
        n (int): 命令编号，默认为 1205。
        """
        json_data = {
            "c": "robot",
            "a": "startCartFeedRot",
            "d": {"direction": direction, "speed": speed},
            "n": n
        }
        return await self.send(json_data)

    async def servo_j(self, joint_positions, n=1206):
        """
        关节运动

        参数:

        joint_positions (list): 关节位置
        n (int): 命令编号，默认为 1206。
        """
        if isinstance(joint_positions, np.ndarray):
            # 如果是numpy数组，则创建一个新数组，在前面加上六个0
            joint_positions = joint_positions.tolist()
        joint_positions = [0, 0, 0, 0, 0, 0] + joint_positions
        json_data = {
            "c": "robot",
            "a": "servoJ",
            "d": joint_positions,
            "n": n
        }
        return await self.send(json_data)

    async def servo_c(self, path, speed, n=1207):
        """
        圆弧运动

        参数:
        path (list): 运动路径
        speed (int): 运动速度
        n (int): 命令编号，默认为 1207。
        """
        json_data = {
            "c": "robot",
            "a": "servoC",
            "d": {"path": path, "speed": speed},
            "n": n
        }
        return await self.send(json_data)

    async def move_list(self, points, speed, n=1208):
        """
        多点运动

        参数:
        points (list): 运动点列表
        speed (int): 运动速度
        n (int): 命令编号，默认为 1208。
        """
        json_data = {
            "c": "robot",
            "a": "moveList",
            "d": {"points": points, "speed": speed},
            "n": n
        }
        return await self.send(json_data)

    async def set_jc_do(self, state, n=1209):
        """
        机器人IO输出设置

        参数:
        state (bool): IO输出状态
        n (int): 命令编号，默认为 1209。
        """
        json_data = {
            "c": "robot",
            "a": "setJCDO",
            "d": state,
            "n": n
        }
        return await self.send(json_data)

    async def set_jc_ai_ctrl(self, n=1210):
        """
        采集机器人IO电路的电流电压值

        参数:
        n (int): 命令编号，默认为 1210。
        """
        json_data = {
            "c": "robot",
            "a": "setJCAICtrl",
            "d": 0,
            "n": n
        }
        return await self.send(json_data)

    async def set_tb_do(self, state, n=1211):
        """
        工具IO输出

        参数:
        state (bool): IO输出状态
        n (int): 命令编号，默认为 1211。
        """
        json_data = {
            "c": "robot",
            "a": "setTBDO",
            "d": state,
            "n": n
        }
        return await self.send(json_data)

    async def set_jc_ao_ctrl(self, n=1212):
        """
        模拟输出IO类型

        参数:
        n (int): 命令编号，默认为 1212。
        """
        json_data = {
            "c": "robot",
            "a": "setJCAOCtrl",
            "d": 0,
            "n": n
        }
        return await self.send(json_data)

    async def set_jc_ao(self, value, n=1213):
        """
        模拟输出IO值

        参数:
        value (int): 输出值
        n (int): 命令编号，默认为 1213。
        """
        json_data = {
            "c": "robot",
            "a": "setJCAO",
            "d": value,
            "n": n
        }
        return await self.send(json_data)

    async def set_tb_dire_vol(self, voltage, n=1214):
        """
        工具IO输出电压

        参数:
        voltage (int): 输出电压
        n (int): 命令编号，默认为 1214。
        """
        json_data = {
            "c": "robot",
            "a": "setTBDireVol",
            "d": voltage,
            "n": n
        }
        return await self.send(json_data)

    async def set_tb_ai_ctrl(self, n=1215):
        """
        工具IO采集控制

        参数:
        n (int): 命令编号，默认为 1215。
        """
        json_data = {
            "c": "robot",
            "a": "setTBAICtrl",
            "d": 0,
            "n": n
        }
        return await self.send(json_data)

    async def get_tb_act_vol(self, n=1216):
        """
        获取TB输出电压

        参数:
        n (int): 命令编号，默认为 1216。
        """
        json_data = {
            "c": "robot",
            "a": "getTBActlVol",
            "d": 0,
            "n": n
        }
        return await self.send(json_data)

    async def get_jc_di(self, n=1217):
        """
        获取JC的数字输入

        参数:
        n (int): 命令编号，默认为 1217。
        """
        json_data = {
            "c": "robot",
            "a": "getJCDI",
            "d": 0,
            "n": n
        }
        return await self.send(json_data)

    async def get_jc_do(self, n=1218):
        """
        获取数字输出

        参数:
        n (int): 命令编号，默认为 1218。
        """
        json_data = {
            "c": "robot",
            "a": "getJCDO",
            "d": 0,
            "n": n
        }
        return await self.send(json_data)

    async def on_changed(self, n=1219):
        """
        机器人信息发生改变时服务端主动上报的信息

        参数:
        n (int): 命令编号，默认为 1219。
        """
        json_data = {
            "c": "robot",
            "a": "OnChanged",
            "d": 0,
            "n": n
        }
        return await self.send(json_data)

    async def joint2cart(self, joints, n=1220):
        """
        关节轴信息转换成坐标信息

        参数:
        joints (list): 关节轴信息
        n (int): 命令编号，默认为 1220。
        """
        json_data = {
            "c": "robot",
            "a": "joint2cart",
            "d": joints,
            "n": n
        }
        return await self.send(json_data)

    async def cart2joint(self, expect, carts, equip=0, n=1014):
        """
        坐标信息转换成关节轴信息

        参数:
        expect (list): 期望的关节位置。
        carts (list): 直线位置。
        equip (int): 设备编号，默认为 0。
        n (int): 命令编号，默认为 1014。
        """
        json_data = {
            "c": "robot",
            "a": "cart2joint",
            "d": {
                "equip": equip,
                "expect": expect,
                "carts": carts
            },
            "n": n
        }
        return await self.send(json_data)

    async def work_coord_make(self, work, key, value, n=1222):
        """
        创建机器人工作坐标系

        参数:
        work (str): 工作坐标系
        key (str): 键
        value (list): 值
        n (int): 命令编号，默认为 1222。
        """
        json_data = {
            "c": "robot",
            "a": "workCoordMake",
            "d": {"work": work, "key": key, "value": value},
            "n": n
        }
        return await self.send(json_data)

    async def tool_coord_org(self, tool, key, value, n=1223):
        """
        工具坐标系原点校准

        参数:
        tool (str): 工具坐标系
        key (str): 键
        value (list): 值
        n (int): 命令编号，默认为 1223。
        """
        json_data = {
            "c": "robot",
            "a": "toolCoordOrg",
            "d": {"tool": tool, "key": key, "value": value},
            "n": n
        }
        return await self.send(json_data)

    async def tool_coord_dir(self, tool, key, value, n=1224):
        """
        工具坐标系方向校准

        参数:
        tool (str): 工具坐标系
        key (str): 键
        value (list): 值
        n (int): 命令编号，默认为 1224。
        """
        json_data = {
            "c": "robot",
            "a": "toolCoordDir",
            "d": {"tool": tool, "key": key, "value": value},
            "n": n
        }
        return await self.send(json_data)

    async def get_euler_offset(self, point, n=1225):
        """
        相对点

        参数:
        point (list): 参考点
        n (int): 命令编号，默认为 1225。
        """
        json_data = {
            "c": "robot",
            "a": "getEulerOffset",
            "d": point,
            "n": n
        }
        return await self.send(json_data)

    async def euler2rot_mat(self, euler, n=1226):
        """
        转换TCP（质心转换）

        参数:
        euler (list): 欧拉角
        n (int): 命令编号，默认为 1226。
        """
        json_data = {
            "c": "robot",
            "a": "euler2rotMat",
            "d": euler,
            "n": n
        }
        return await self.send(json_data)

    async def move_line_and_wait(self, target_geom, tolerance=0.05, timeout=30):
        """
        发送直线移动指令，并阻塞直到机械臂到达目标位置或超时。
        target_geom 应包含目标位置的 x, y, z 坐标。
        tolerance 是接受的误差范围。
        timeout 是等待的最长时间，以秒为单位。
        """
        await self.move_line(target_geom)
        start_time = time.time()
        while True:
            current_time = time.time()
            if current_time - start_time > timeout:
                print("Timeout reached, movement may not have completed.")
                break
            response = await self.websocket.recv()
            response = self.wait_for_send(response)
            print("response:", response)
            if 'a' in response:
                continue
            if response['n'] == 1128 and response['d'] == True:
                break
            await asyncio.sleep(0.01)  # 检查间隔，防止频繁查询

