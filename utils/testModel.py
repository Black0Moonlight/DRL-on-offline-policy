import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from envs.rl_reach_env import RLReachEnv
from algo.DDPG import DDPGAgent
from algo.DDPG_Prior import DDPGAgent3
from algo.SAC import SACAgent
from config import *
from utils.ReplayMemory import *
import os

np.set_printoptions(precision=3, suppress=True)  # 设定numpy打印精度
np.random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)

env = RLReachEnv(is_render=True, is_good_view=False)


def clearLogs():
    # 清理掉上一次训练存储的模型及日志
    ls = os.listdir(save_path)
    for i in ls:
        os.remove(os.path.join(save_path, i))
    # ls = os.listdir(load_path)
    # for i in ls:
    #     os.remove(os.path.join(save_path, i))
    ls = os.listdir(log_path)
    for i in ls:
        os.remove(os.path.join(log_path, i))


def testNet(net_num, state_dim, action_dim, net):
    if net == "DDPG":
        Agent1 = DDPGAgent(state_dim, action_dim, action_bound=1)
    else:
        Agent1 = SACAgent(state_dim, action_dim, action_bound=1)
    if net_num > 0:
        print('/***************** Load Pretrain NO.{} network ******************/'.format(net_num))
        Agent1.load_net(net_num)
    score = 0
    for i in range(200):
        if action_dim == 3:
            s0 = env.reset_xyz()
        else:
            s0 = env.reset_joint()
        done = is_success = 0
        # time.sleep(0.3)
        for j in range(opt.max_steps_one_episode):
            if done:
                break
            a0 = Agent1.select_action(s0)
            if action_dim == 3:
                s1, r, done, is_success = env.step_xyz(a0)
            else:
                s1, r, done, is_success = env.step_joint(a0)
            s0 = s1
            # time.sleep(0.1)
        score += is_success
        # time.sleep(0.3)
    print('/***************** NO.{} network {:.1f}% ******************/'.format(net_num, score/2))
    return net_num, score/2


def xyz_DDPG(state_dim, action_dim, batch_size, action_bound=1):
    print("/************************* train xyz **************************/")
    # 使用ReplayMemory，可以使用sample和一条在线策略的on sample
    replay_buffer = ReplayMemory(opt.buffer_size)
    action_bound = 0.3 + float(env.action_space.high[0])
    Agent = DDPGAgent(state_dim, action_dim, action_bound=action_bound)

    last_total_r = 0
    save_num = 0
    time_start = t0 = time.perf_counter()
    for episode in range(opt.max_episodes_xyz):
        s0 = env.reset_xyz()
        total_r = total_l = loss = 0
        done = 0
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
            if replay_buffer.__len__() > batch_size:
                loss = Agent.update(replay_buffer.sample(batch_size))
                total_l += loss
        writer.add_scalar('Reward', total_r, episode)
        writer.add_scalar('Loss', total_l, episode)
        print("/*********************** Episode {} End ***********************/".format(episode))
        print('Total time:{}s Total Reward:{} dr:{}'
              .format(time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - time_start)), total_r,
                      total_r - last_total_r))
        print()
        last_total_r = total_r
        if (episode+1) % 10 == 0:
            save_num += 1
            Agent.save(save_num)
            net_num, d = testNet(save_num, state_dim, action_dim, "DDPG")
            writer.add_scalar('Success_rate', d, save_num)
            # net_num, d = testandtrainNet(save_num, state_dim, action_dim, "DDPG")
            # writer.add_scalar('testand train', d, net_num)


def train_xyz_SAC(state_dim, action_dim, batch_size, action_bound=1):
    print("/************************* train xyz **************************/")
    action_bound = 0.3 + float(env.action_space.high[0])

    replay_buffer = ReplayMemory(opt.buffer_size)
    Agent = SACAgent(state_dim, action_dim, batch_size=batch_size, action_bound=action_bound)

    last_total_r = 0
    save_num = 0
    time_start = t0 = time.perf_counter()
    for episode in range(opt.max_episodes_xyz):
        s0 = env.reset_xyz()
        total_r = total_l = loss = 0
        done = 0
        i = 0
        for i in range(opt.max_steps_one_episode):
            if done:
                break
            a0 = Agent.select_action(s0)
            a0 = (a0 + np.random.normal(0, 0.2, size=action_dim))
            s1, r, done, is_success = env.step_xyz(a0)
            replay_buffer.push(s0, a0, r, s1, 1 - done)
            s0 = s1
            total_r += r
            # time.sleep(0.1)
            # if i > 1 and i % 50 == 0:
            #     t1 = time.perf_counter()
            #     dt = t1 - t0
            #     t0 = t1
            #     print('Episode {} Step {} dt:{:.1f}s Reward:{} loss:{}'
            #           .format(episode, i, dt, r, loss))
            if replay_buffer.__len__() > Agent.batch_size:
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
        if (episode+1) % 10 == 0:
            save_num += 1
            Agent.save(save_num)
            net_num, d = testNet(save_num, state_dim, action_dim, "SAC")
            writer.add_scalar('Success_rate', d, save_num)


def xyz_DDPG_OnOff(state_dim, action_dim, batch_size, action_bound=1):
    print("/************************* train xyz **************************/")
    replay_buffer = ReplayMemory1(opt.buffer_size)
    action_bound = 0.3 + float(env.action_space.high[0])
    Agent = DDPGAgent(state_dim, action_dim, action_bound=action_bound)

    last_total_r = 0
    save_num = 0
    time_start = t0 = time.perf_counter()
    for episode in range(opt.max_episodes_xyz):
        s0 = env.reset_xyz()
        total_r = total_l = loss = 0
        done = 0
        i = 0
        for i in range(opt.max_steps_one_episode):
            if done:
                break
            a0 = Agent.select_action(s0)
            a0 = (a0 + np.random.normal(0, 0.2, size=action_dim))
            s1, r, done, is_success = env.step_xyz(a0)
            replay_buffer.push(s0, a0, r, s1, 1 - done)
            s0 = s1
            total_r += r
            if replay_buffer.__len__() > batch_size and replay_buffer.update:
                loss = Agent.update(replay_buffer.on_sample(batch_size))
                total_l += loss
        writer.add_scalar('Reward', total_r, episode)
        writer.add_scalar('Loss', total_l, episode)
        print("/*********************** Episode {} End ***********************/".format(episode))
        print('Total time:{}s Total Reward:{} dr:{}'
              .format(time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - time_start)), total_r,
                      total_r - last_total_r))
        print()
        last_total_r = total_r
        if (episode+1) % 10 == 0:
            save_num += 1
            Agent.save(save_num)
            net_num, d = testNet(save_num, state_dim, action_dim, "DDPG")
            writer.add_scalar('Success_rate', d, save_num)
            # net_num, d = testandtrainNet(save_num, state_dim, action_dim, "DDPG")
            # writer.add_scalar('testand train', d, net_num)


def xyz_DDPG_Prioritized\
                (state_dim, action_dim, batch_size, action_bound=1):
    print("/************************* train xyz **************************/")
    replay_buffer = PrioritizedReplayBuffer(opt.buffer_size)
    action_bound = 0.3 + float(env.action_space.high[0])
    Agent = DDPGAgent3(state_dim, action_dim, action_bound=action_bound)

    last_total_r = 0
    save_num = 0
    time_start = t0 = time.perf_counter()
    for episode in range(opt.max_episodes_xyz):
        s0 = env.reset_xyz()
        total_r = total_l = loss = 0
        done = 0
        i = 0
        for i in range(opt.max_steps_one_episode):
            if done:
                break
            a0 = Agent.select_action(s0)
            a0 = (a0 + np.random.normal(0, 0.2, size=action_dim))
            s1, r, done, is_success = env.step_xyz(a0)
            replay_buffer.push(s0, a0, r, s1, 1 - done)
            s0 = s1
            total_r += r
            if replay_buffer.__len__() > batch_size and replay_buffer.update:
                indices, batch, is_weights = replay_buffer.on_sample(batch_size)
                loss = Agent.update(indices, batch, is_weights)
                total_l += loss
        writer.add_scalar('Reward', total_r, episode)
        writer.add_scalar('Loss', total_l, episode)
        print("/*********************** Episode {} End ***********************/".format(episode))
        print('Total time:{}s Total Reward:{} dr:{}'
              .format(time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - time_start)), total_r,
                      total_r - last_total_r))
        print()
        last_total_r = total_r
        if (episode+1) % 10 == 0:
            save_num += 1
            Agent.save(save_num)
            net_num, d = testNet(save_num, state_dim, action_dim, "DDPG")
            writer.add_scalar('Success_rate', d, save_num)
            # net_num, d = testandtrainNet(save_num, state_dim, action_dim, "DDPG")
            # writer.add_scalar('testand train', d, net_num)


'''###################################################
注意检查训练7自由度固定点和随机点时的验证频率是否一致

###################################################'''

def joint_DDPG(state_dim, action_dim, batch_size, action_bound=1):
    print("/************************* train joint ************************/")
    replay_buffer = ReplayMemory(opt.buffer_size)
    Agent = DDPGAgent(state_dim, action_dim, action_bound=action_bound)
    last_total_r = 0
    save_num = 0
    time_start = t0 = time.perf_counter()
    for episode in range(opt.max_episodes):
        s0 = env.reset_joint()
        total_r = total_l = loss = 0
        done = 0
        i = 0
        for i in range(opt.max_steps_one_episode):
            if done:
                break
            a0 = Agent.select_action(s0)
            a0 += np.random.normal(0, 0.2, size=action_dim)
            s1, r, done, is_success = env.step_joint(a0)
            replay_buffer.push(s0, a0, r, s1, 1 - done)
            s0 = s1
            total_r += r
            # time.sleep(0.1)
            # if i > 1 and i % 50 == 0:
            #     t1 = time.perf_counter()
            #     dt = t1 - t0
            #     t0 = t1
            #     print('Episode {} Step {} dt:{:.1f}s Reward:{} loss:{}'
            #           .format(episode, i, dt, r, loss))
            if replay_buffer.__len__() > batch_size:
                loss = Agent.update(replay_buffer.on_sample(batch_size))
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
            net_num, d = testNet(save_num, state_dim, action_dim, "DDPG")
            writer.add_scalar('Success rate', d, save_num)


def joint_DDPG_OnOff(state_dim, action_dim, batch_size, action_bound=1):
    print("/************************* train joint ************************/")
    # 使用ReplayMemory1
    replay_buffer = ReplayMemory1(opt.buffer_size)
    Agent = DDPGAgent(state_dim, action_dim, action_bound=action_bound)
    last_total_r = 0
    save_num = 0
    time_start = t0 = time.perf_counter()
    for episode in range(opt.max_episodes):
        s0 = env.reset_joint()
        total_r = total_l = loss = 0
        done = 0
        i = 0
        for i in range(opt.max_steps_one_episode):
            if done:
                break
            a0 = Agent.select_action(s0)
            a0 += np.random.normal(0, 0.2, size=action_dim)
            s1, r, done, is_success = env.step_joint(a0)
            replay_buffer.push(s0, a0, r, s1, 1 - done)
            s0 = s1
            total_r += r
            if replay_buffer.__len__() > batch_size and replay_buffer.update:
                loss = Agent.update(replay_buffer.on_sample(batch_size))
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
            net_num, d = testNet(save_num, state_dim, action_dim, "DDPG")
            writer.add_scalar('Success rate', d, save_num)


def joint_DDPG_Prioritized\
                (state_dim, action_dim, batch_size, action_bound=1):
    print("/************************* train joint ************************/")
    # 使用PrioritizedReplayBuffer
    replay_buffer = PrioritizedReplayBuffer(opt.buffer_size)
    Agent = DDPGAgent3(state_dim, action_dim, action_bound=action_bound)
    last_total_r = 0
    save_num = 0
    time_start = t0 = time.perf_counter()
    for episode in range(opt.max_episodes):
        s0 = env.reset_joint()
        total_r = total_l = loss = 0
        done = 0
        i = 0
        for i in range(opt.max_steps_one_episode):
            if done:
                break
            a0 = Agent.select_action(s0)
            a0 += np.random.normal(0, 0.2, size=action_dim)
            s1, r, done, is_success = env.step_joint(a0)
            replay_buffer.push(s0, a0, r, s1, 1 - done)
            s0 = s1
            total_r += r
            if replay_buffer.__len__() > batch_size and replay_buffer.update:
                indices, batch, is_weights = replay_buffer.on_sample(batch_size)
                loss = Agent.update(indices, batch, is_weights)
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
            net_num, d = testNet(save_num, state_dim, action_dim, "DDPG")
            writer.add_scalar('Success rate', d, save_num)


if __name__ == '__main__':
    clearLogs()
    writer = SummaryWriter(log_path)  # 展示部分数据

    # testNet(1, 6, 3, "DDPG")

    # xyz_DDPG(state_dim=6, action_dim=3, batch_size=16)
    # xyz_DDPG_OnOff(state_dim=6, action_dim=3, batch_size=16)
    # xyz_DDPG_Prioritized(state_dim=6, action_dim=3, batch_size=16)

    # train_xyz_SAC(state_dim=6, action_dim=3, batch_size=16)
    #
    # joint_DDPG(state_dim=13, action_dim=7, batch_size=16)
    joint_DDPG_OnOff(state_dim=13, action_dim=7, batch_size=16)
    # joint_DDPG_Prioritized(state_dim=13, action_dim=7, batch_size=16)

