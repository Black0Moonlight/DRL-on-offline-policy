import numpy as np
import vrep.sim as sim
from config import opt


class ArmEnv(object):
    print('Program started')
    sim.simxFinish(-1)  # just in case, close all opened connections
    clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to CoppeliaSim
    if clientID != -1:
        print('Connected to remote API server')
    else:
        print('Failed connecting to remote API server')

    action_bound = [-1, 1]
    goal_xyz = {'x': 0.85, 'y': 0.05, 'z': 0.90, 'r': 0.12}  # 蓝色目标区域的 x,y 坐标和球半径r
    goal_orient = 25 * np.pi / 180

    state_dim = 15
    action_dim = 6

    distance_old = 0
    distance_new = 0
    orient_old = 0
    orient_new = 0
    all_steps = 0

    # 时间步计数器
    step_counter = 0

    init_joint_positions = [0, 0, 0, -88.5 * np.pi / 180, 0, 89.4 * np.pi / 180, 0]

    # 索取句柄
    arm_joint = {}
    _, arm_joint[0] = sim.simxGetObjectHandle(clientID, 'LBR_iiwa_7_R800_joint1', sim.simx_opmode_blocking)
    _, arm_joint[1] = sim.simxGetObjectHandle(clientID, 'LBR_iiwa_7_R800_joint2', sim.simx_opmode_blocking)
    _, arm_joint[2] = sim.simxGetObjectHandle(clientID, 'LBR_iiwa_7_R800_joint3', sim.simx_opmode_blocking)
    _, arm_joint[3] = sim.simxGetObjectHandle(clientID, 'LBR_iiwa_7_R800_joint4', sim.simx_opmode_blocking)
    _, arm_joint[4] = sim.simxGetObjectHandle(clientID, 'LBR_iiwa_7_R800_joint5', sim.simx_opmode_blocking)
    _, arm_joint[5] = sim.simxGetObjectHandle(clientID, 'LBR_iiwa_7_R800_joint6', sim.simx_opmode_blocking)
    _, arm_joint[6] = sim.simxGetObjectHandle(clientID, 'LBR_iiwa_7_R800_joint7', sim.simx_opmode_blocking)
    _, goal = sim.simxGetObjectHandle(clientID, 'goal', sim.simx_opmode_blocking)
    _, tip = sim.simxGetObjectHandle(clientID, 'tip', sim.simx_opmode_blocking)

    def __init__(self):  # 初始化
        # 初始姿态
        self.arm_info = [0, 0, 0, -88.5*np.pi/180, 0, 89.4*np.pi/180]
        self.robot_joint_positions = self.init_joint_positions

        self.max_steps_one_episode = opt.max_steps_one_episode

        # 获取目标姿态
        _, self.orient_goal = sim.simxGetObjectOrientation(self.clientID, self.goal, -1, sim.simx_opmode_blocking)
        self.orient_goal_gamma = self.orient_goal[2]
        self.on_goal = 0
        self.on_goal2 = 0
        # 距离值的normalization,手抓中心(1.65,-0.28,0.944),最远抓取点(0.85,0.33,0.79),也就是第六个纱锭前方
        self.dist_norm = 2*np.sqrt((1.65-0.85)**2+(-0.28-0.33)**2+(0.944-0.79)**2)  # 1.018*2

    def clip_val(self, value, limit):
        if value < limit[0]:
            return limit[0]
        if value > limit[1]:
            return limit[1]
        return value

    def step_joint(self, action):
        # 机械臂移动范围限制
        self.x_low_obs = self.goal_xyz['x']-0.2
        self.x_high_obs = self.goal_xyz['x']+0.2
        self.y_low_obs = self.goal_xyz['y']-0.2
        self.y_high_obs = self.goal_xyz['y']+0.2
        self.z_low_obs = self.goal_xyz['z']-0.2
        self.z_high_obs = self.goal_xyz['y']+0.2

        self.terminated = False
        self.is_success = False

        action = action * np.pi / 180

        # 登记各个节点的信息
        numJoints = 7
        sim.simxPauseCommunication(self.clientID, 1)
        for i in range(numJoints):
            # _, current_joint[i] = sim.simxGetJointPosition(self.clientID, self.arm_joint[i], sim.simx_opmode_oneshot)
            action[i] = self.clip_val(action[i], self.action_bound)
            self.robot_joint_positions[i] += action[i]
            sim.simxSetJointPosition(self.clientID, self.arm_joint[i], self.robot_joint_positions[i], sim.simx_opmode_oneshot)
        sim.simxPauseCommunication(self.clientID, 0)
        sim.simxGetPingTime(self.clientID)  # 必须有

        # 获取手抓中心坐标和姿态
        _, finger_xyz = sim.simxGetObjectPosition(self.clientID, self.tip, -1, sim.simx_opmode_blocking)  # 手抓中心坐标
        _, finger_orient = sim.simxGetObjectOrientation(self.clientID, self.tip, self.goal, sim.simx_opmode_blocking)

        # 计算末端点与目标区域的距离
        goal_pos = np.array([self.goal_xyz['x'], self.goal_xyz['y'], self.goal_xyz['z']])  # 目标位置(x、y、z)
        self.distance_new = np.linalg.norm((finger_xyz - goal_pos) / self.dist_norm, axis=-1)

        # r1 以末端点到目标点的距离
        r = -self.distance_new

        # r2
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
        x = finger_xyz[0]
        y = finger_xyz[1]
        z = finger_xyz[2]
        terminated = bool(x < self.x_low_obs or x > self.x_high_obs
                          or y < self.y_low_obs or y > self.y_high_obs
                          or z < self.z_low_obs or z > self.z_high_obs)
        if terminated:
            r += -0.05

        self.step_counter += 1

        return np.concatenate((finger_xyz, self.robot_joint_positions, goal_pos)), r, self.terminated, self.is_success

    def step2(self, action):
        done2 = False
        # 目标区域动态调整
        self.all_steps += 1
        if self.all_steps >= 60000:
            self.goal_xyz['r'] = 0.07
        if self.all_steps >= 90000:
            self.goal_xyz['r'] = 0.04

        self.goal_xyz['x'] = 0.72
        # self.goal_xyz['r'] = 0.04
        goal_pos = np.array([self.goal_xyz['x'], self.goal_xyz['y'], self.goal_xyz['z']])  # 目标位置(x、y、z)
        sim.simxSetObjectPosition(self.clientID, self.goal, -1, goal_pos, sim.simx_opmode_oneshot)
        sim.simxGetPingTime(self.clientID)  # 必须有,否则可能不执行

        action = action * np.pi / 180
        self.arm_info = self.arm_info + action

        # 控制命令需要同时方式，故暂停通信，用于存储所有控制命令一起发送
        sim.simxPauseCommunication(self.clientID, 1)
        for i in range(6):
            sim.simxSetJointPosition(self.clientID, self.arm_joint[i], self.arm_info[i], sim.simx_opmode_oneshot)
        sim.simxPauseCommunication(self.clientID, 0)
        sim.simxGetPingTime(self.clientID)  # 必须有

        # 获取手抓中心坐标和姿态
        # sim.simxPauseCommunication(self.clientID, 1)
        _, finger_xyz = sim.simxGetObjectPosition(self.clientID, self.tip, -1, sim.simx_opmode_blocking)  # 手抓中心坐标
        _, finger_orient = sim.simxGetObjectOrientation(self.clientID, self.tip, self.goal, sim.simx_opmode_blocking)
        # sim.simxPauseCommunication(self.clientID, 0)
        # sim.simxGetPingTime(self.clientID)  # 必须有

        # 计算末端点与目标区域的距离，并转换到[0，1]之间,(0.85-0.72)*2=0.26
        dist1 = [(self.goal_xyz['x'] - finger_xyz[0]) / self.dist_norm,
                 (self.goal_xyz['y'] - finger_xyz[1]) / self.dist_norm,
                 (self.goal_xyz['z'] - finger_xyz[2]) / self.dist_norm]  # (delta x, delta y, delta z)

        # 以goal坐标系为基准，绕z轴旋转改变γ值。故α值任意，α/β要与goal保持一致(为0)。orient(α,β,γ)
        # α,β,γ范围[-pi,pi],正则化到[-1,1]
        dist2 = [finger_orient[0] / np.pi, finger_orient[1] / np.pi]

        # r1 以末端点到目标点的距离(转换到0~1之间),alpha/beta差距平均值作为单步奖励
        r2 = -np.sqrt(dist1[0] ** 2 + dist1[1] ** 2 + dist1[2] ** 2) - 0.25*(np.abs(dist2[0]) + np.abs(dist2[1]))
        # r2
        self.distance_new = np.sqrt(dist1[0] ** 2 + dist1[1] ** 2 + dist1[2] ** 2)
        if self.distance_new < self.distance_old:
            r2 += 0.005
        else:
            r2 -= 0.005
        self.distance_old = self.distance_new.copy()
        # r3
        self.orient_new = np.abs(dist2[0]) + np.abs(dist2[1])
        if self.orient_new < self.orient_old:
            r2 += 0.004
        else:
            r2 -= 0.004
        self.orient_old = self.orient_new.copy()

        # 连续停留50步则认为完成(done),给出奖励
        if self.goal_xyz['x'] - self.goal_xyz['r'] / 2 < finger_xyz[0] < self.goal_xyz['x'] + self.goal_xyz['r'] / 2:
            if self.goal_xyz['y'] - self.goal_xyz['r'] / 2 < finger_xyz[1] < self.goal_xyz['y'] + \
                    self.goal_xyz['r'] / 2:
                if self.goal_xyz['z'] - self.goal_xyz['r'] / 2 < finger_xyz[2] < self.goal_xyz['z'] + \
                        self.goal_xyz['r'] / 2:
                    r2 += 1.
                    self.on_goal2 += 1
                    if self.on_goal2 > 50:
                        done2 = True
        else:
            self.on_goal2 = 0

        # state：关节1至关节6角度，tip的坐标(finger_xyz)，tip到goal的距离(dist1)，goal到tip的β、γ(dist2),on_goal
        s2 = np.concatenate((self.arm_info, np.array([finger_xyz[0], finger_xyz[1], finger_xyz[2]]) / self.dist_norm,
                            dist1, dist2, [1. if self.on_goal else 0.]))  # dim:6+3+3+2+1=15

        return s2, r2, done2

    def reset_joint(self):
        self.step_counter = 0
        self.terminated = False

        self.goal_xyz['x'] = 1.65
        self.goal_xyz['y'] = -0.5
        self.goal_xyz['z'] = 0.85
        goal_pos = np.array([self.goal_xyz['x'], self.goal_xyz['y'], self.goal_xyz['z']])  # 目标位置(x、y、z)
        self.robot_joint_positions = [0, 0, 0, -88.5 * np.pi / 180, 0, 89.4 * np.pi / 180, 0]  # self.init_joint_positions

        # 初始化iiwa姿态
        sim.simxPauseCommunication(self.clientID, 1)
        for k in range(7):
            sim.simxSetJointPosition(self.clientID, self.arm_joint[k], self.robot_joint_positions[k], sim.simx_opmode_oneshot)
        sim.simxSetObjectPosition(self.clientID, self.goal, -1, goal_pos, sim.simx_opmode_oneshot)
        sim.simxPauseCommunication(self.clientID, 0)
        sim.simxGetPingTime(self.clientID)  # 必须有

        # 获取手抓中心坐标和姿态
        _, finger_xyz = sim.simxGetObjectPosition(self.clientID, self.tip, -1, sim.simx_opmode_blocking)  # 手抓中心坐标
        _, finger_orient = sim.simxGetObjectOrientation(self.clientID, self.tip, self.goal, sim.simx_opmode_blocking)

        return np.concatenate((finger_xyz, self.robot_joint_positions, goal_pos))

    def reset2(self):
        # 刷新时，将目标区域设为随机位置，将on_goal恢复为0
        self.goal_xyz['x'] = 0.84
        self.goal_xyz['y'] = 0.1
        self.goal_xyz['z'] = 0.85
        goal_pos = np.array([self.goal_xyz['x'], self.goal_xyz['y'], self.goal_xyz['z']])  # 目标位置(x、y、z)
        self.on_goal = 0
        self.arm_info = [0, 0, 0, -88.5 * np.pi / 180, 0, 89.4 * np.pi / 180]

        # 初始化iiwa姿态
        sim.simxPauseCommunication(self.clientID, 1)
        for k in range(6):
            sim.simxSetJointPosition(self.clientID, self.arm_joint[k], self.arm_info[k], sim.simx_opmode_oneshot)
        sim.simxSetObjectPosition(self.clientID, self.goal, -1, goal_pos, sim.simx_opmode_oneshot)
        sim.simxPauseCommunication(self.clientID, 0)
        sim.simxGetPingTime(self.clientID)  # 必须有

        # 获取手抓中心坐标和姿态
        # sim.simxPauseCommunication(self.clientID, 1)
        _, finger_xyz = sim.simxGetObjectPosition(self.clientID, self.tip, -1, sim.simx_opmode_blocking)  # 手抓中心坐标
        _, finger_orient = sim.simxGetObjectOrientation(self.clientID, self.tip, self.goal, sim.simx_opmode_blocking)
        # sim.simxPauseCommunication(self.clientID, 0)
        # sim.simxGetPingTime(self.clientID)  # 必须有

        # 计算末端点与目标区域的距离，并转换到[0，1]之间
        dist1 = [(self.goal_xyz['x'] - finger_xyz[0]) / self.dist_norm,
                 (self.goal_xyz['y'] - finger_xyz[1]) / self.dist_norm,
                 (self.goal_xyz['z'] - finger_xyz[2]) / self.dist_norm]    # (delta x, delta y, delta z)

        # 以goal坐标系为基准，绕z轴旋转改变α值。故γ值任意，α/β要与goal保持一致(为0)。orient(α,β,γ)
        # α,β,γ范围[-pi,pi],正则化到[-1,1]
        dist2 = [finger_orient[0]/np.pi, finger_orient[1]/np.pi]

        self.distance_old = np.sqrt(dist1[0] ** 2 + dist1[1] ** 2 + dist1[2] ** 2)  # 末端与目标点距离
        self.orient_old = np.abs(dist2[0]) + np.abs(dist2[1])

        # state：关节1至关节6角度，tip的坐标(finger_xyz)，tip到goal的距离(dist1)，goal到tip的β、γ(dist2),on_goal
        s = np.concatenate((self.arm_info, np.array([finger_xyz[0], finger_xyz[1], finger_xyz[2]])/self.dist_norm,
                            dist1, dist2, [1. if self.on_goal else 0.]))  # dim:6+3+3+2+1=15
        return s

    def sample_action(self):
        return np.random.rand(6) - 0.5  # two radians


if __name__ == '__main__':
    env = ArmEnv()
    while True:
        env.step(np.random.rand(6))
