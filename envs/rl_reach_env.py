import numpy as np
import pybullet as p
import pybullet_data
import os
from gym import spaces
import random
import time
import math
from config import opt


class RLReachEnv(object):
    """创建强化学习机械臂reach任务仿真环境"""
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    distance_old = 0  # 夹爪与物体距离
    distance_new = 0

    num_joints = 0
    robot_joint_pos = [0, 0, 0, 0, 0, 0, 0]

    robot_grip_pos = [0, 0, 0]  # 机械臂末端坐标
    object_pos = [0, 0, 0]  # 物体坐标

    def __init__(self, is_render=False, is_good_view=False):
        """
            is_render (bool):       是否创建场景可视化
            is_good_view (bool):    是否创建更优视角
        """

        self.kuka_id = None
        self.object_id = None

        self.terminated = None
        self.is_success = None

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

        self.dist_norm = 2 * np.sqrt((1.65 - 0.85) ** 2 + (-0.28 - 0.33) ** 2 + (0.944 - 0.79) ** 2)

        # 设置相机
        p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                     cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])

        # 动作空间
        self.action_space = spaces.Box(
            low=np.array([self.x_low_action, self.y_low_action, self.z_low_action]),
            high=np.array([self.x_high_action, self.y_high_action, self.z_high_action]))

        # 状态空间
        self.observation_space = spaces.Box(
            low=np.array([self.x_low_obs, self.y_low_obs, self.z_low_obs,
                          self.x_low_obs, self.y_low_obs, self.z_low_obs]),
            high=np.array([self.x_high_obs, self.y_high_obs, self.z_high_obs,
                           self.x_high_obs, self.y_high_obs, self.z_high_obs]))

        # 计数器
        self.step_counter = 0

        self.urdf_root_path = pybullet_data.getDataPath()
        # lower limits for null space
        self.lower_limits = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # upper limits for null space
        self.upper_limits = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # joint ranges for null space
        self.joint_ranges = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # restposes for null space
        self.rest_poses = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        # joint damping coefficents
        self.joint_damping = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001
        ]

        # 初始关节角度
        self.init_joint_positions = [
            0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684,
            -0.006539
        ]

        self.orientation = p.getQuaternionFromEuler(
            [0., -math.pi, math.pi / 2.])

    def reset_xyz(self):
        # 初始化计数器
        self.step_counter = 0
        # 运行结束标志
        self.terminated = False
        self.is_success = False

        p.resetSimulation()
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        # 初始化重力
        p.setGravity(0, 0, -10)

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
        # 载入机械臂
        self.kuka_id = p.loadURDF(os.path.join(self.urdf_root_path, "kuka_iiwa/model.urdf"), useFixedBase=True)
        # 载入桌子
        p.loadURDF(os.path.join(self.urdf_root_path, "table/table.urdf"), basePosition=[0.5, 0, -0.65])
        # 载入物体
        self.object_pos[0] = random.uniform(self.x_low_obs, self.x_high_obs)
        self.object_pos[1] = random.uniform(self.y_low_obs, self.y_high_obs)
        self.object_pos[2] = random.uniform(self.z_low_obs, self.z_high_obs)
        # self.obj_pos[0] = (self.x_low_obs+self.x_high_obs) / 2
        # self.obj_pos[1] = (self.y_low_obs+self.y_high_obs) / 2
        # self.obj_pos[2] = (self.z_low_obs + self.z_high_obs) / 2
        ang = 3.14 * 0.5 + 3.1415925438 * random.random()
        orn = p.getQuaternionFromEuler([0, 0, ang])
        self.object_id = p.loadURDF("../models/cube_small_target_push.urdf",
                                    basePosition=self.object_pos,
                                    baseOrientation=orn,
                                    useFixedBase=1)
        # 关节角初始化
        self.num_joints = p.getNumJoints(self.kuka_id)
        for i in range(self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.init_joint_positions[i],
            )
        # 机器臂末端xyz
        self.robot_grip_pos = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        p.stepSimulation()
        # self.object_pos = p.getBasePositionAndOrientation(self.object_id)[0]

        self.object_pos = np.array(
            p.getBasePositionAndOrientation(self.object_id)[0]).astype(
            np.float32)
        # state：机械臂末端*3 物体位置*3 = 6
        return np.concatenate((self.robot_grip_pos, self.object_pos))
        # return np.hstack((np.array(self.robot_grip_pos).astype(np.float32), self.object_pos))

    def reset_joint(self):
        # 初始化计数器
        self.step_counter = 0
        # 运行结束标志
        self.terminated = False
        self.is_success = False

        p.resetSimulation()
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        # 初始化重力
        p.setGravity(0, 0, -10)

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
        # 载入机械臂
        self.kuka_id = p.loadURDF(os.path.join(self.urdf_root_path, "kuka_iiwa/model.urdf"), useFixedBase=True)
        # 载入桌子
        p.loadURDF(os.path.join(self.urdf_root_path, "table/table.urdf"), basePosition=[0.5, 0, -0.65])
        # 载入物体
        self.object_pos[0] = random.uniform(self.x_low_obs, self.x_high_obs)
        self.object_pos[1] = random.uniform(self.y_low_obs, self.y_high_obs)
        self.object_pos[2] = random.uniform(self.z_low_obs, self.z_high_obs)
        # self.obj_pos[0] = (self.x_low_obs+self.x_high_obs) / 2
        # self.obj_pos[1] = (self.y_low_obs+self.y_high_obs) / 2
        # self.obj_pos[2] = (self.z_low_obs + self.z_high_obs) / 2
        ang = 3.14 * 0.5 + 3.1415925438 * random.random()
        orn = p.getQuaternionFromEuler([0, 0, ang])
        self.object_id = p.loadURDF("../models/cube_small_target_push.urdf",
                                    basePosition=self.object_pos,
                                    baseOrientation=orn,
                                    useFixedBase=1)
        # 关节角初始化
        self.num_joints = p.getNumJoints(self.kuka_id)
        for i in range(self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.init_joint_positions[i],
            )
        # 机器臂末端xyz
        self.robot_grip_pos = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        p.stepSimulation()
        # self.object_pos = p.getBasePositionAndOrientation(self.object_id)[0]

        # goal = [random.uniform(self.x_low_obs, self.x_high_obs),
        #         random.uniform(self.y_low_obs, self.y_high_obs),
        #         random.uniform(self.z_low_obs, self.z_high_obs)]
        self.object_pos = np.array(
            p.getBasePositionAndOrientation(self.object_id)[0]).astype(
            np.float32)
        # state：机械臂末端*3, 关节角度*7, 物体位置*3 = 13
        return np.concatenate((self.robot_grip_pos, self.init_joint_positions, self.object_pos))
        # return np.hstack((np.array(self.robot_grip_pos).astype(np.float32), self.object_pos))

    def clip_val(self, value, limit):
        if value < limit[0]:
            return limit[0]
        if value > limit[1]:
            return limit[1]
        return value

    def reward(self):
        """根据state计算当前的reward"""
        # 获取机械臂当前的末端坐标
        # 一定注意是取第4个值，请参考pybullet手册的这个函数返回值的说明
        self.robot_grip_pos = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        # self.object_state=p.getBasePositionAndOrientation(self.object_id)
        # self.object_state=np.array(self.object_state).astype(np.float32)

        # 获取物体当前的位置坐标
        self.object_pos = np.array(
            p.getBasePositionAndOrientation(self.object_id)[0]).astype(
                np.float32)

        # r1 用机械臂末端和物体的距离作为奖励函数的依据
        self.distance_new = np.linalg.norm((self.robot_grip_pos - self.object_pos) / self.dist_norm, axis=-1)
        r = -self.distance_new

        # r2 如果运动趋势正确，提高奖励
        if self.distance_new < self.distance_old:
            r += 0.5
        elif self.distance_new > self.distance_old:
            r -= 0.5

        # r3 如果机械臂一直无所事事，在最大步数还不能接触到物体，也需要给一定的惩罚
        if self.step_counter > self.max_steps_one_episode:
            r += -20
            self.terminated = True
            self.is_success = False
        elif self.distance_new < opt.reach_dis:
            r += 10
            self.terminated = True
            self.is_success = True

        # r4 如果机械比末端超过了obs的空间，也视为done，给予一定的惩罚
        x = self.robot_grip_pos[0]
        y = self.robot_grip_pos[1]
        z = self.robot_grip_pos[2]
        terminated = bool(x < self.x_low_obs or x > self.x_high_obs
                          or y < self.y_low_obs or y > self.y_high_obs
                          or z < self.z_low_obs or z > self.z_high_obs)
        if terminated:
            r += -0.05
            # self.terminated = True
            # self.is_success = False

        self.distance_old = self.distance_new
        self.step_counter += 1

        # 在代码开始部分，如果定义了is_good_view，那么机械臂的动作会变慢，方便观察
        if self.is_good_view:
            time.sleep(0.05)

        return np.concatenate((self.robot_grip_pos, self.robot_joint_pos, self.object_pos)),\
            r, self.terminated, self.is_success

    def step_xyz(self, action):
        """限定机械臂末端全局运动范围"""
        limit_x = [0.2, 0.7]
        limit_y = [-0.3, 0.3]
        limit_z = [0, 0.55]

        dv = opt.reach_ctr
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv

        # 获取当前机械臂末端坐标
        self.robot_grip_pos = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        # 计算下一步的机械臂末端坐标
        self.robot_grip_pos = [
            self.clip_val(self.robot_grip_pos[0] + dx, limit_x), self.clip_val(self.robot_grip_pos[1] + dy, limit_y),
            self.clip_val(self.robot_grip_pos[2] + dz, limit_z)
        ]
        # 通过逆运动学计算机械臂移动到新位置的关节角度
        self.robot_joint_pos = p.calculateInverseKinematics(
            bodyUniqueId=self.kuka_id,
            endEffectorLinkIndex=self.num_joints - 1,
            targetPosition=self.robot_grip_pos,
            targetOrientation=self.orientation,
            jointDamping=self.joint_damping,
        )
        # 使机械臂移动到新位置
        for i in range(self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.robot_joint_pos[i],
            )
        p.stepSimulation()

        # 在代码开始部分，如果定义了is_good_view，那么机械臂的动作会变慢，方便观察
        if self.is_good_view:
            time.sleep(0.05)

        self.step_counter += 1

        """根据state计算当前的reward"""
        # 获取机械臂当前的末端坐标
        # 一定注意是取第4个值，请参考pybullet手册的这个函数返回值的说明
        self.robot_grip_pos = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        # self.object_state=p.getBasePositionAndOrientation(self.object_id)
        # self.object_state=np.array(self.object_state).astype(np.float32)

        # 获取物体当前的位置坐标
        self.object_pos = np.array(
            p.getBasePositionAndOrientation(self.object_id)[0]).astype(
                np.float32)

        # r1 用机械臂末端和物体的距离作为奖励函数的依据
        self.distance_new = np.linalg.norm((self.robot_grip_pos - self.object_pos) / self.dist_norm, axis=-1)
        r = -self.distance_new

        # r2 如果运动趋势正确，提高奖励
        if self.distance_new < self.distance_old:
            r += 0.5
        elif self.distance_new > self.distance_old:
            r -= 0.5

        # r3 如果机械臂一直无所事事，在最大步数还不能接触到物体，也需要给一定的惩罚
        if self.step_counter > self.max_steps_one_episode:
            r += -20
            self.terminated = True
            self.is_success = False
        elif self.distance_new < opt.reach_dis:
            r += 10
            self.terminated = True
            self.is_success = True

        # r4 如果机械比末端超过了obs的空间，也视为done，给予一定的惩罚
        x = self.robot_grip_pos[0]
        y = self.robot_grip_pos[1]
        z = self.robot_grip_pos[2]
        self.terminated = bool(x < self.x_low_obs or x > self.x_high_obs
                          or y < self.y_low_obs or y > self.y_high_obs
                          or z < self.z_low_obs or z > self.z_high_obs)
        # if terminated:
        #     r += -50
        #     self.terminated = True
        #     self.is_success = False

        # info = {'distance:', self.distance_new}

        self.distance_old = self.distance_new

        # goal = [random.uniform(self.x_low_obs,self.x_high_obs),
        #         random.uniform(self.y_low_obs,self.y_high_obs),
        #         random.uniform(self.z_low_obs, self.z_high_obs)]
        return np.hstack((np.array(self.robot_grip_pos).astype(np.float32), self.object_pos)), \
            r, self.terminated, self.is_success

    def step_joint(self, action):
        """限定机械臂关节运动范围"""
        action_bound = [-1.5, 1.5]
        current_joint = [0, 0, 0, 0, 0, 0, 0]

        """限定机械臂末端全局运动范围"""
        limit_x = [0.2, 0.7]
        limit_y = [-0.3, 0.3]
        limit_z = [0, 0.55]

        # 登记各个节点的信息
        numJoints = 7  # p.getNumJoints(self.kuka_id)
        for i in range(numJoints):
            info = p.getJointInfo(self.kuka_id, i)
            # 计算新的角度
            current_joint[i] = p.getJointState(self.kuka_id, i)[0]
            # action[i] = action[i] * opt.reach_ctr
            # action[i] = action[i] * np.pi / 180
            action[i] = self.clip_val(action[i], action_bound)
            self.robot_joint_pos[i] = self.clip_val(current_joint[i] + action[i], [info[8], info[9]])

        # 使机械臂移动到新位置
        for i in range(self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.robot_joint_pos[i],
            )
        p.stepSimulation()

        # 获取当前机械臂末端坐标
        self.robot_grip_pos = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]

        # # 限制当前机械臂末端坐标
        # self.new_grip_pos = [
        #     self.clip_val(self.robot_grip_pos[0], limit_x), self.clip_val(self.robot_grip_pos[1], limit_y),
        #     self.clip_val(self.robot_grip_pos[2], limit_z)
        # ]
        # if not self.robot_grip_pos == self.new_grip_pos:
        #     # 通过逆运动学计算机械臂移动到新位置的关节角度
        #     self.robot_joint_positions = p.calculateInverseKinematics(
        #         bodyUniqueId=self.kuka_id,
        #         endEffectorLinkIndex=self.num_joints - 1,
        #         targetPosition=[self.new_grip_pos[0], self.new_grip_pos[1], self.new_grip_pos[2]],
        #         targetOrientation=self.orientation,
        #         jointDamping=self.joint_damping,
        #     )
        #     # 使机械臂移动到新位置
        #     for i in range(self.num_joints):
        #         p.resetJointState(
        #             bodyUniqueId=self.kuka_id,
        #             jointIndex=i,
        #             targetValue=self.robot_joint_positions[i],
        #         )
        #     p.stepSimulation()
        return self.reward()

    def close(self):
        p.disconnect()


# if __name__ == '__main__':
    # # 这一部分是做baseline，即让机械臂随机选择动作，看看能够得到的分数
    # env = RLReachEnv(is_good_view=True, is_render=True)
    # print('env={}'.format(env))
    # print(env.observation_space.shape)
    # print(env.action_space.shape)
    # obs = env.reset()
    # action = env.action_space.sample()
    # obs, reward, done, info = env.step_xyz(action)
    # print('obs={},reward={},done={}'.format(obs, reward, done))
    #
    # sum_reward = 0
    # success_times = 0
    # for i in range(100):
    #     env.reset()
    #     for j in range(1000):
    #         action = env.action_space.sample()
    #         obs, reward, done, info = env.step_xyz(action)
    #         print('reward={},done={}'.format(reward, done))
    #         sum_reward += reward
    #         if reward == 1:
    #             success_times += 1
    #         if done:
    #             break
    #     # time.sleep(0.1)
    # print()
    # print('sum_reward={}'.format(sum_reward))
    # print('success rate={}'.format(success_times / 50))
