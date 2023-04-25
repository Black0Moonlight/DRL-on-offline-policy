import time
import numpy as np
from envs.rl_reach_env import RLReachEnv
from algo.DDPG import DDPGAgent
from algo.SAC import SACAgent
from config import *
from utils.ReplayMemory import *

np.set_printoptions(precision=3, suppress=True)  # 设定numpy打印精度
np.random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)

env = RLReachEnv(is_render=True, is_good_view=False)


def testNet(net_num, state_dim, action_dim, net):
    if net == "DDPG":
        Agent = DDPGAgent(state_dim, action_dim, action_bound=1)
    else:
        Agent = SACAgent(state_dim, action_dim, action_bound=1)
    if net_num > 0:
        print('/***************** Load Pretrain NO.{} network ******************/'.format(net_num))
        Agent.load_net(net_num)
    score = 0
    for i in range(1000):
        if action_dim == 3:
            s0 = env.reset_xyz()
        else:
            s0 = env.reset_joint()
        done = is_success = 0
        # time.sleep(0.3)
        for j in range(opt.max_steps_one_episode):
            if done:
                break
            a0 = Agent.select_action(s0)
            if action_dim == 3:
                s1, r, done, is_success = env.step_xyz(a0)
            else:
                s1, r, done, is_success = env.step_joint(a0)
            s0 = s1
            time.sleep(0.1)
        score += is_success
        # time.sleep(0.3)
    print('/***************** NO.{} network {:.1f}% ******************/'.format(net_num, score/10))
    return net_num, score/1000


def train_xyz_SAC(state_dim, action_dim, batch_size, action_bound=1):
    print("/************************* train xyz **************************/")

    action_bound = 0.3 + float(env.action_space.high[0])
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

        print("/*********************** Episode {} End ***********************/".format(episode))
        print('Total time:{}s Total Reward:{} dr:{}'
              .format(time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - time_start)), total_r,
                      total_r - last_total_r))
        print()
        last_total_r = total_r
        if (episode + 1) % 50 == 0:
            save_num += 1
            Agent.save(save_num)
    net_num, d = testNet(save_num, state_dim, action_dim, "SAC")


if __name__ == '__main__':
    # testNet(1, 6, 3, "DDPG")
    train_xyz_SAC(state_dim=6, action_dim=3, batch_size=16)
