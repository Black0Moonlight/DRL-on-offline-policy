import os
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from envs.rl_reach_env import RLReachEnv
# from vrep.iiwa7 import ArmEnv
from algo.DDPG import DDPGAgent
from algo.SAC import SACAgent
from config import *

np.set_printoptions(precision=3, suppress=True)  # 设定numpy打印精度
np.random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)

# MODE = 'train_xyz'
# MODE = 'train_joint'
# MODE = 'test'

env = RLReachEnv(is_render=True, is_good_view=False)
# iiwa7_env = ArmEnv()

def clearLogs():
    # 清理掉上一次训练存储的模型及日志
    ls = os.listdir(save_path)
    for i in ls:
        os.remove(os.path.join(save_path, i))
    ls = os.listdir(log_path)
    for i in ls:
        os.remove(os.path.join(log_path, i))


clearLogs()

writer = SummaryWriter(log_path)  # 展示部分数据
# file = open(csv_path, 'a', encoding='utf-8', newline='')  # 'a' 追加数据
# csv_writer = csv.writer(file)  # 储存全部数据

def testNet(net_num=1):
    Agent = SACAgent(6, 3, action_bound=1)
    if net_num > 0:
        print('/***************** Load Pretrain NO.{} network ******************/'.format(net_num))
        Agent.load_net(net_num)
    d = 0
    for i in range(100):
        s0 = env.reset_xyz()
        done = is_success = 0
        # time.sleep(0.3)
        for j in range(opt.max_steps_one_episode):
            if done:
                break
            a0 = Agent.select_action(s0)
            s1, r, done, is_success = env.step_xyz(a0)
            s0 = s1
            # time.sleep(0.01)
        # print(d, is_success)
        d += is_success
        # time.sleep(0.3)
    print('/***************** NO.{} network {:.1f}% ******************/'.format(net_num, d))
    return net_num, d


def train_joint_SAC(state_dim, action_dim, action_bound=1):
    print("/************************* train joint ************************/")
    writer = SummaryWriter(log_path)  # 展示部分数据
    time_start = time.perf_counter()
    Agent = SACAgent(state_dim, action_dim, action_bound=action_bound)
    last_total_r = 0
    save_num = 0
    for episode in range(opt.max_episodes):
        s0 = env.reset_joint()
        total_r = total_l = 0
        done = 0
        for i in range(opt.max_steps_one_episode):
            if done:
                break
            a0 = Agent.select_action(s0)
            a0 += np.random.normal(0, 0.4, size=action_dim)
            s1, r, done, is_success = env.step_joint(a0)
            Agent.put(s0, a0, r, s1, 1 - done)

            s0 = s1
            total_r += r
            if Agent.replay_buffer.__len__() > Agent.batch_size:
                loss = Agent.update()
                total_l += loss
        writer.add_scalar('Reward', total_r, episode)
        writer.add_scalar('Loss', total_l, episode)
        print("/*********************** Episode {} End ***********************/".format(episode))
        print('Total time:{}s Total Reward:{} dr:{}'
              .format(time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - time_start)), total_r,
                      total_r - last_total_r))
        print()
        last_total_r = total_r
        if episode > 1 and episode % 50 == 0:
            Agent.save(save_num)
            save_num += 1
    writer.close()

def train_joint_DDPG(state_dim, action_dim, action_bound=1):
    print("/************************* train joint ************************/")

    Agent = DDPGAgent(state_dim, action_dim, batch_size=1024, action_bound=action_bound)
    # Agent.load_net(96)
    last_total_r = 0
    save_num = 0
    time_start = t0 = time.perf_counter()
    for episode in range(opt.max_episodes):
        s0 = env.reset_joint()
        total_r = 0
        done = 0
        loss = 0
        total_l = 0
        i = 0
        for i in range(opt.max_steps_one_episode):
            if done:
                break
            a0 = Agent.select_action(s0)
            a0 += np.random.normal(0, 0.8, size=action_dim)
            s1, r, done, is_success = env.step_joint(a0)
            Agent.put(s0, a0, r, s1, 1 - done)
            s0 = s1
            total_r += r
            # time.sleep(0.01)
            if Agent.replay_buffer.__len__() > Agent.batch_size:
                loss = Agent.update()
                total_l += loss
        writer.add_scalar('Reward', total_r, episode)
        writer.add_scalar('Loss', total_l, episode)
        print("/*********************** Episode {} End ***********************/".format(episode))
        print('Total time:{}s Total Reward:{} dr:{}'
              .format(time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - time_start)), total_r,
                      total_r - last_total_r))
        print()
        last_total_r = total_r
        if (episode+1) % 200 == 0:
            save_num += 1
            Agent.save(save_num)

def train_xyz_SAC(state_dim, action_dim, action_bound=1):
    print("/************************* train xyz **************************/")

    action_bound = 0.3 + float(env.action_space.high[0])
    Agent = SACAgent(state_dim, action_dim, action_bound=action_bound)

    last_total_r = 0
    save_num = 0
    time_start = t0 = time.perf_counter()
    for episode in range(opt.max_episodes_xyz):
        s0 = env.reset_xyz()
        total_r = 0
        done = 0
        loss = 0
        total_l = 0
        i = 0
        for i in range(opt.max_steps_one_episode):
            if done:
                break
            a0 = Agent.select_action(s0)
            a0 = (a0 + np.random.normal(0, 0.2, size=action_dim))
            s1, r, done, is_success = env.step_xyz(a0)
            Agent.put(s0, a0, r, s1, 1 - done)
            s0 = s1
            total_r += r
            # time.sleep(0.1)
            # if i > 1 and i % 50 == 0:
            #     t1 = time.perf_counter()
            #     dt = t1 - t0
            #     t0 = t1
            #     print('Episode {} Step {} dt:{:.1f}s Reward:{} loss:{}'
            #           .format(episode, i, dt, r, loss))
            if Agent.replay_buffer.__len__() > Agent.batch_size:
                loss = Agent.update()
                total_l += loss
        writer.add_scalar('Reward', total_r, episode)
        writer.add_scalar('Loss', total_l, episode)
        print("/*********************** Episode {} End ***********************/".format(episode))
        print('Total time:{}s Total Reward:{} dr:{}'
              .format(time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - time_start)), total_r,
                      total_r - last_total_r))
        print()
        last_total_r = total_r
        if (episode + 1) % 50 == 0:
            save_num += 1
            Agent.save(save_num)
    net_num, d = testNet(save_num)
    writer.add_scalar('Success rate', d, net_num)

def train_xyz_DDPG(state_dim, action_dim, action_bound=1):
    print("/************************* train xyz **************************/")

    action_bound = 0.3 + float(env.action_space.high[0])
    Agent = DDPGAgent(state_dim, action_dim, action_bound=action_bound)

    last_total_r = 0
    save_num = 0
    time_start = t0 = time.perf_counter()
    for episode in range(opt.max_episodes_xyz):
        s0 = env.reset_xyz()
        total_r = 0
        done = 0
        loss = 0
        total_l = 0
        i = 0
        for i in range(opt.max_steps_one_episode):
            if done:
                break
            a0 = Agent.select_action(s0)
            a0 = (a0 + np.random.normal(0, 0.2, size=action_dim))
            s1, r, done, is_success = env.step_xyz(a0)
            Agent.put(s0, a0, r, s1, 1 - done)
            s0 = s1
            total_r += r
            # time.sleep(0.1)
            # if i > 1 and i % 50 == 0:
            #     t1 = time.perf_counter()
            #     dt = t1 - t0
            #     t0 = t1
            #     print('Episode {} Step {} dt:{:.1f}s Reward:{} loss:{}'
            #           .format(episode, i, dt, r, loss))
            if Agent.replay_buffer.__len__() > Agent.batch_size:
                loss = Agent.update()
                total_l += loss
        writer.add_scalar('Reward', total_r, episode)
        writer.add_scalar('Loss', total_l, episode)
        print("/*********************** Episode {} End ***********************/".format(episode))
        print('Total time:{}s Total Reward:{} dr:{}'
              .format(time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - time_start)), total_r,
                      total_r - last_total_r))
        print()
        last_total_r = total_r
        if (episode+1) % 50 == 0:
            save_num += 1
            Agent.save(save_num)
    net_num, d = testNet(save_num)
    writer.add_scalar('Success rate', d, net_num)


if __name__ == '__main__':
    # testNet(4)
    train_joint_DDPG(state_dim=13, action_dim=7)
    #train_joint_SAC(state_dim=13, action_dim=7)
    #train_xyz_DDPG(state_dim=6, action_dim=3)
    #train_xyz_SAC(state_dim=6, action_dim=3)
    print("/***************************** End ****************************/")
    writer.close()

