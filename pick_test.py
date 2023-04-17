import numpy as np
import pybullet as p
import pybullet_data
import os
from gym import spaces
import random
import time
import math
from config import *


class RLPickEnv(object):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    distance_old = 0  # 夹爪与最终目标距离
    distance_new = 0
    distance_object = 0  # 夹爪与物体距离
    distance_target = 0  # 物体与目标位置距离

    num_joints = 0
    robot_joint_pos = np.zeros(14)
    grip_joint_pos = [-0.3, 0.3]

    robot_grip_pos = [0, 0, 0]  # 机械臂末端坐标
    object_pos = [0, 0, 0]  # 物体坐标
    target_pos = [0, 0, 0]  # 目标位置坐标

    robot_grip_orn = [0, 0, 0]  # 机械臂末端坐标
    object_orn = [0, 0, 0]  # 物体坐标
    target_orn = [0, 0, 0]  # 目标位置坐标
    orn_error = 0

    last_object_pos = 0.0
    last_target_pos = 0.0
    current_object_pos = 0.0
    current_target_pos = 0.0

    def __init__(self, is_render=False, is_good_view=False):
        """
            is_render (bool):       是否创建场景可视化
            is_good_view (bool):    是否创建更优视角
        """

        self.sphere_id = None
        self.kuka_id = None
        self.object_id = None
        self.target_object_id = None

        self.catchFlag = False
        self.terminated = False
        self.is_success = False

        self.is_render = is_render
        self.is_good_view = is_good_view
        self.max_steps_one_episode = opt.max_steps_one_episode

        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        # 机械臂移动范围限制
        self.x_low_obs = 0.21
        self.x_high_obs = 0.69
        self.y_low_obs = -0.29
        self.y_high_obs = 0.29
        self.z_low_obs = 0.1
        self.z_high_obs = 0.54

        # 机械臂动作范围限制
        self.x_low_action = -0.4
        self.x_high_action = 0.4
        self.y_low_action = -0.4
        self.y_high_action = 0.4
        self.z_low_action = -0.4
        self.z_high_action = 0.4

        self.end_effector_index = 6
        self.gripper_index = 7
        self.gripper_length = 0.257

        # 设置相机
        p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                     cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])

        # 动作空间
        self.action_space = spaces.Box(
            low=np.array([self.x_low_action, self.y_low_action, self.z_low_action]),
            high=np.array([self.x_high_action, self.y_high_action, self.z_high_action]),
            dtype=np.float32)

        # 状态空间
        self.observation_space = spaces.Box(
            low=np.array([self.x_low_obs, self.y_low_obs, self.z_low_obs + self.gripper_length]),
            high=np.array([self.x_high_obs, self.y_high_obs, self.z_high_obs + self.gripper_length]),
            dtype=np.float32)

        # 时间步计数器
        self.step_counter = 0

        self.urdf_root_path = pybullet_data.getDataPath()
        # joint damping coefficents
        self.joint_damping = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001
        ]

        # 初始关节角度
        self.init_joint_positions = [
            0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539,
            0.000048, -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
        ]

        self.orientation = p.getQuaternionFromEuler(
            [0., -math.pi, math.pi / 2.])

        self.reset()

    def reset(self):
        # 初始化时间步计数器
        self.step_counter = 0
        # 重置运行结束标志
        self.catchFlag = False
        self.terminated = False
        self.is_success = False

        self.grip_joint_pos = [-0.3, 0.3]

        p.resetSimulation()
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        # 初始化重力
        p.setGravity(0, 0, -9.8)

        # 状态空间的限制空间可视化，以白线标识
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, 0],
            lineToXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_high_obs, 0],
            lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_low_obs, 0],
            lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_high_obs, 0],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])

        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])

        # 载入平面
        p.loadURDF(os.path.join(self.urdf_root_path, "plane.urdf"), basePosition=[0, 0, -0.65])
        # 载入机械臂 带夹爪
        self.kuka_id = p.loadSDF(os.path.join(self.urdf_root_path, "kuka_iiwa/kuka_with_gripper2.sdf"))[0]
        for i in range(self.num_joints):
            p.enableJointForceTorqueSensor(bodyUniqueId=self.kuka_id, jointIndex=i, enableSensor=1)
            # p.changeDynamics(bodyUniqueId=self.kuka_id,
            #                  linkIndex=i,
            #                  contactStiffness=10000,
            #                  contactDamping=0.8,)
        # 载入桌子
        table_uid = p.loadURDF(os.path.join(self.urdf_root_path, "table/table.urdf"), basePosition=[0.5, 0, -0.65])
        p.changeVisualShape(table_uid, -1, rgbaColor=[1, 1, 1, 1])

        # 通过循环筛选出满足距离条件的随机点
        for _ in range(1000):
            # 物体的位姿
            self.object_pos[0] = random.uniform(self.x_low_obs, self.x_high_obs)
            self.object_pos[1] = random.uniform(self.y_low_obs, self.y_high_obs)
            self.object_pos[2] = 0.01   # TODO 原z=0.01
            self.object_pos[0] = (self.x_low_obs+self.x_high_obs) / 2
            self.object_pos[1] = (self.y_low_obs+self.y_high_obs) / 2
            self.object_pos[2] = 0.01

            # 目标物体的位姿
            self.target_pos[0] = random.uniform(self.x_low_obs, self.x_high_obs)
            self.target_pos[1] = random.uniform(self.y_low_obs, self.y_high_obs)
            self.target_pos[2] = random.uniform(self.z_low_obs, self.z_high_obs)  # TODO 原z=0.01
            self.target_pos[0] = (self.x_low_obs+self.x_high_obs) / 2 + 0.15
            self.target_pos[1] = (self.y_low_obs+self.y_high_obs) / 2 - 0.15
            self.target_pos[2] = 0.01

            # 确保距离在一定范围内
            self.dis_between_target_block = np.linalg.norm((np.array(self.object_pos) - np.array(self.target_pos)), axis=-1)
            if self.dis_between_target_block >= 0.22 and self.dis_between_target_block <= 0.25:
                break

        ang = 3.14 * 0.5 + 3.1415925438 * random.random()
        ang = 0
        orn = p.getQuaternionFromEuler([0, 0, ang])

        ang_target = 3.14 * 0.5 + 3.1415925438 * random.random()
        ang_target = 0
        orn_target = p.getQuaternionFromEuler([0, 0, ang_target])

        # 载入物体
        self.object_id = p.loadURDF("../models/cube_small_pick.urdf",
                                    basePosition=self.object_pos,
                                    baseOrientation=orn)
        p.changeDynamics(bodyUniqueId=self.object_id,
                         linkIndex=-1,
                         contactStiffness=10000.0,
                         contactDamping=0.1)
        # Set the debug color of the block to red
        p.setDebugObjectColor(self.object_id, -1, objectDebugColorRGB=[1, 0, 0])
        # enable the debug visualization
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)

        # 获取当前机械臂末端坐标
        self.robot_grip_pos = np.array(p.getLinkState(self.kuka_id, self.end_effector_index)[4]).astype(np.float32)
        self.sphere_id = p.loadURDF("../models/sphere.urdf",
                                 basePosition=[self.robot_grip_pos[0], self.robot_grip_pos[1],
                                               self.robot_grip_pos[2] - self.gripper_length],
                                 useFixedBase=1)
        # 载入目标物体
        self.target_object_id = p.loadURDF("../models/cube_small_target_pick.urdf",
                                    basePosition=self.target_pos,
                                    baseOrientation=orn_target,
                                    useFixedBase=1)
        # 避免碰撞检测
        p.setCollisionFilterPair(self.target_object_id, self.object_id, -1, -1, 0)
        p.setCollisionFilterPair(self.sphere_id, self.object_id, -1, -1, 0)
        p.setCollisionFilterPair(self.target_object_id, self.sphere_id, -1, -1, 0)

        # 关节角初始化
        self.num_joints = p.getNumJoints(self.kuka_id)
        for i in range(self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.init_joint_positions[i],
            )
        p.resetJointState(self.kuka_id, 8, self.grip_joint_pos[0])
        p.resetJointState(self.kuka_id, 10, 0)
        p.resetJointState(self.kuka_id, 11, self.grip_joint_pos[1])
        p.resetJointState(self.kuka_id, 13, 0)
        # for i in range(self.num_joints):
        #      print(p.getJointInfo(self.kuka_id, i))

        self.robot_grip_pos = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        # logging.debug("init_pos={}\n".format(p.getLinkState(self.kuka_id,self.num_joints-1)))
        p.stepSimulation()

        # state = np.hstack((self.robot_grip_pos, self.object_pos, self.target_pos))
        state = np.hstack((self.robot_grip_pos, self.robot_grip_orn[2],
                           self.object_pos, self.object_orn[2],
                           self.target_pos))

        return state

    def step_xyz(self, action):

        # 通过逆运动学计算机械臂移动到新位置的关节角度
        self.robot_joint_pos = np.array(p.calculateInverseKinematics(
            bodyUniqueId=self.kuka_id,
            endEffectorLinkIndex=self.end_effector_index,
            targetPosition=action,
            targetOrientation=self.orientation,  # p.getQuaternionFromEuler([0., -math.pi, self.robot_grip_orn[2]])
            jointDamping=self.joint_damping,
        ))
        # 使机械臂移动到新位置
        for i in range(self.end_effector_index):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.robot_joint_pos[i],
            )
        p.resetJointState(self.kuka_id, self.end_effector_index, self.robot_grip_orn[2])
        p.resetBasePositionAndOrientation(bodyUniqueId=self.sphere_id,
                                          posObj=[self.robot_grip_pos[0],
                                                  self.robot_grip_pos[1],
                                                  self.robot_grip_pos[2] - self.gripper_length],
                                          ornObj=self.orientation)
        p.stepSimulation()

    def grip(self, catch=0):
        if catch == OPEN:
            for i in range(25):
                self.grip_joint_pos[0] -= 0.01
                self.grip_joint_pos[1] += 0.01
                p.resetJointState(self.kuka_id, 7, 0)
                p.resetJointState(self.kuka_id, 8, self.grip_joint_pos[0])
                p.resetJointState(self.kuka_id, 9, 0)
                p.resetJointState(self.kuka_id, 10, 0)
                p.resetJointState(self.kuka_id, 11, self.grip_joint_pos[1])
                p.resetJointState(self.kuka_id, 12, 0)
                p.resetJointState(self.kuka_id, 13, 0)
                p.stepSimulation()
                time.sleep(0.1)
        elif catch == CLOSE:
            for i in range(25):
                self.grip_joint_pos[0] += 0.01
                self.grip_joint_pos[1] -= 0.01
                p.resetJointState(self.kuka_id, 7, 0)
                p.resetJointState(self.kuka_id, 8, self.grip_joint_pos[0])
                p.resetJointState(self.kuka_id, 9, 0)
                p.resetJointState(self.kuka_id, 10, 0)
                p.resetJointState(self.kuka_id, 11, self.grip_joint_pos[1])
                p.resetJointState(self.kuka_id, 12, 0)
                p.resetJointState(self.kuka_id, 13, 0)
                p.stepSimulation()
                time.sleep(0.1)
        else:
            p.resetJointState(self.kuka_id, 7, 0)
            p.resetJointState(self.kuka_id, 8, self.grip_joint_pos[0])
            p.resetJointState(self.kuka_id, 9, 0)
            p.resetJointState(self.kuka_id, 10, 0)
            p.resetJointState(self.kuka_id, 11, self.grip_joint_pos[1])
            p.resetJointState(self.kuka_id, 12, 0)
            p.resetJointState(self.kuka_id, 13, 0)
            p.stepSimulation()


if __name__ == '__main__':
    env = RLPickEnv(is_good_view=True, is_render=True)
    env.step_xyz([])
    for i in range(100):
        env.test_grip(i/(25*np.pi))
        time.sleep(0.1)
    time.sleep(1)
    for i in range(100):
        env.test_grip((100-i)/(25*np.pi))
        time.sleep(0.1)
    time.sleep(1)


