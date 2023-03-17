from torch.utils.tensorboard import SummaryWriter
from envs.rl_pick_env import *
from envs.rl_pick_env import RLPickEnv
from algo.SAC import SACAgent
from config import *

np.set_printoptions(precision=3, suppress=True)  # 设定numpy打印精度
np.random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)

# MODE = 'train'
# MODE = 'train_joint'
# MODE = 'test'

env = RLPickEnv(is_render=True, is_good_view=False)


def clearLogs():
    # 清理掉上一次训练存储的模型及日志
    ls = os.listdir(save_path)
    for i in ls:
        os.remove(os.path.join(save_path, i))
    ls = os.listdir(log_path)
    for i in ls:
        os.remove(os.path.join(log_path, i))


def train_xyz(state_dim, action_dim):
    print("/************************* train xyz **************************/")

    action_bound = float(env.action_space.high[0])
    writer = SummaryWriter(log_path)
    Agent = SACAgent(state_dim, action_dim, action_bound=action_bound)

    last_total_r = 0
    save_num = 0
    time_start = t0 = time.perf_counter()
    for episode in range(opt.max_episodes):
        s0 = env.reset()
        total_r = 0
        done = 0
        loss = 0
        for i in range(opt.max_steps_pick):
            if done:
                break
            a0 = Agent.select_action(s0)
            a0 = (a0 + np.random.normal(0, 0.2, size=action_dim))
            s1, r, done, is_success = env.step_xyz(a0)
            Agent.put(s0, a0, r, s1, 1 - done)
            s0 = s1
            total_r += r
            # time.sleep(0.1)
            s1[2] -= env.gripper_length
            distance_object = np.linalg.norm(s1[0:3] - s1[6:9], axis=-1)
            if Agent.replay_buffer.__len__() > Agent.batch_size:
                loss = Agent.update()
            # if i > 1 and i % 50 == 0:
            #     t1 = time.perf_counter()
            #     dt = t1 - t0
            #     t0 = t1
            #     print('Episode {} Step {} dt:{:.1f}s Reward:{:.5f} loss:{:.5f} dist:{:.5f}'
            #           .format(episode, i, dt, r, loss, distance_object))
        writer.add_scalar('Reward', total_r, episode)
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


if __name__ == '__main__':
    # if MODE == 'test':
    #     Agent = SACAgent(9, 3, action_bound=1)
    #     net_num = 31
    #     if net_num > 0:
    #         print('/***************** Load Pretrain NO.{} network ******************/'.format(net_num))
    #         Agent.load_net(net_num)
    #     for i in range(5):
    #         s0 = env.reset()
    #         time.sleep(0.3)
    #         for i in range(opt.max_steps_one_episode):
    #             a0 = Agent.select_action(s0)
    #             s1, r, done, is_success = env.step_xyz(a0)
    #             s0 = s1
    #             time.sleep(0.01)
    #         time.sleep(0.3)
    # else:
    train_xyz(state_dim=11, action_dim=4)
    print("/***************************** End ****************************/")

