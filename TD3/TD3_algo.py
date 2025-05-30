"""
Note: This is a updated version from my previous code,
for the target network, I use moving average to soft replace target parameters instead using assign function.
By doing this, it has 20% speed up on my machine (CPU).

Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.

Using:
tensorflow 1.14.0
gym 0.15.3
"""

import tensorflow as tf
import numpy as np
from UAV_env import UAVEnv
import time
import matplotlib.pyplot as plt
from state_normalization import StateNormalization

#####################  hyper parameters  ####################
#MAX_EPISODES = 1000
MAX_EPISODES = 50000

LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
# LR_A = 0.1  # learning rate for actor
# LR_C = 0.2  # learning rate for critic
# GAMMA = 0.001  # optimal reward discount
GAMMA = 0.001  # reward discount
TAU = 0.01  # soft replacement
VAR_MIN = 0.01  # 最小高斯噪声
# MEMORY_CAPACITY = 5000
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
OUTPUT_GRAPH = False


# XXX: 公平性指标JFI
def calculate_jfi(values):
    values = np.array(values)
    numerator = np.sum(values) ** 2
    denominator = len(values) * np.sum(values ** 2)
    return numerator / denominator if denominator != 0 else 0

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(ddpg, eval_episodes=10):
    # eval_env = gym.make(env_name)
    eval_env = UAVEnv()
    # eval_env.seed(seed + 100)
    avg_reward = 0.
    for i in range(eval_episodes):
        state = eval_env.reset()
        # while not done:
        for j in range(int(len(eval_env.UE_loc_list))):
            action = ddpg.choose_action(state)
            action = np.clip(action, *a_bound)
            state, reward = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


###############################  TD3  ####################################
class TD3:
    def __init__(self, a_dim, s_dim, a_bound):
        self.sess = tf.Session()
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0

        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.A = tf.placeholder(tf.float32, [None, a_dim], 'a')

        with tf.variable_scope('Actor'):
            self.a = self._build_actor(self.S, 'eval', True)
            a_ = self._build_actor(self.S_, 'target', False)

        with tf.variable_scope('Critic1'):
            q1 = self._build_critic(self.S, self.A, 'eval1', True)
            q1_ = self._build_critic(self.S_, a_, 'target1', False)

        with tf.variable_scope('Critic2'):
            q2 = self._build_critic(self.S, self.A, 'eval2', True)
            q2_ = self._build_critic(self.S_, a_, 'target2', False)

        self.actor_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.actor_target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.critic1_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic1/eval1')
        self.critic1_target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic1/target1')
        self.critic2_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic2/eval2')
        self.critic2_target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic2/target2')

        noise = tf.clip_by_value(tf.random.normal(tf.shape(a_), stddev=0.2), -0.5, 0.5)
        a_ = tf.clip_by_value(a_ + noise, -a_bound[1], a_bound[1])
        q_target = self.R + GAMMA * tf.minimum(q1_, q2_)

        td_error1 = tf.losses.mean_squared_error(q_target, q1)
        td_error2 = tf.losses.mean_squared_error(q_target, q2)
        self.ctrain1 = tf.train.AdamOptimizer(LR_C).minimize(td_error1, var_list=self.critic1_params)
        self.ctrain2 = tf.train.AdamOptimizer(LR_C).minimize(td_error2, var_list=self.critic2_params)

        self.a_loss = -tf.reduce_mean(self._build_critic(self.S, self.a, 'eval1', True))
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(self.a_loss, var_list=self.actor_params)

        self.soft_update = [tf.assign(t, (1 - TAU) * t + TAU * e) for t, e in
                            zip(self.actor_target_params + self.critic1_target_params + self.critic2_target_params,
                                self.actor_params + self.critic1_params + self.critic2_params)]

        self.sess.run(tf.global_variables_initializer())

    def _build_actor(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 400, tf.nn.relu, trainable=trainable)
            net = tf.layers.dense(net, 300, tf.nn.relu, trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, tf.nn.tanh, trainable=trainable)
            return tf.multiply(a, self.a_bound[1])

    def _build_critic(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.concat([s, a], axis=1)
            net = tf.layers.dense(net, 400, tf.nn.relu, trainable=trainable)
            net = tf.layers.dense(net, 300, tf.nn.relu, trainable=trainable)
            return tf.layers.dense(net, 1, trainable=trainable)

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def store_transition(self, s, a, r, s_):
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = np.hstack((s, a, [r], s_))
        self.pointer += 1

    def learn(self, delay_step):
        indices = np.random.choice(min(self.pointer, MEMORY_CAPACITY), BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run([self.ctrain1, self.ctrain2], {self.S: bs, self.A: ba, self.R: br, self.S_: bs_})
        if delay_step % 2 == 0:
            self.sess.run(self.atrain, {self.S: bs})
            self.sess.run(self.soft_update)

###############################  training  ####################################
np.random.seed(1)
tf.set_random_seed(1)

env = UAVEnv()
MAX_EP_STEPS = env.slot_num
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound  # [-1,1]

td3 = TD3(a_dim, s_dim, a_bound)

#var = 0.1  # control exploration
var = 0.01  # control exploration 噪声方差
t1 = time.time()
ep_reward_list = []
JFI_list = []
s_normal = StateNormalization()

for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    # XXX: 新增公平性指标（被服务次数）
    ep_jfi = np.zeros(4)
    j = 0
    while j < MAX_EP_STEPS:
        # Add exploration noise
        a = td3.choose_action(s_normal.state_normal(s))
        # 高斯噪声add randomness to action selection for exploration
        a = np.clip(np.random.normal(a, var), *a_bound)  
        s_, r, is_terminal, step_redo, offloading_ratio_change, reset_dist, ue_id = env.step(a)
        if step_redo:
            continue
        if reset_dist:
            a[2] = -1
        if offloading_ratio_change:
            a[3] = -1
        # 训练奖励缩小10倍
        td3.store_transition(s_normal.state_normal(s), a, r, s_normal.state_normal(s_))  

        if td3.pointer > MEMORY_CAPACITY:
            # 噪声衰减
            #var = max([var * 0.9995, VAR_MIN])  # decay the action randomness
            td3.learn(delay_step=i)
        s = s_
        ep_reward += r
        ep_jfi[ue_id] = ep_jfi[ue_id] + 1
        if j == MAX_EP_STEPS - 1 or is_terminal:
            print('Episode:', i, ' Steps: %2d' % j, ' Reward: %7.2f' % ep_reward, 'Explore: %.3f' % var)
            ep_reward_list = np.append(ep_reward_list, ep_reward)
            JFI = calculate_jfi(ep_jfi)
            JFI_list = np.append(JFI_list, JFI)
            # file_name = 'output_ddpg_' + str(self.bandwidth_nums) + 'MHz.txt'
            file_name = 'output.txt'
            with open(file_name, 'a') as file_obj:
                file_obj.write("\n======== This episode is done ========")  # 本episode结束
            break
        j = j + 1

    # # Evaluate episode
    # if (i + 1) % 50 == 0:
    #     eval_policy(ddpg, env)

print('Running time: ', time.time() - t1)
plt.plot(ep_reward_list)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

jfi_window = 100
# jfi_window = 10
episodes = np.arange(0, MAX_EPISODES)
epi_downsampled = episodes[::jfi_window]
jfi_downsampled = np.mean(JFI_list.reshape(-1, jfi_window), axis=1)
plt.plot(epi_downsampled, jfi_downsampled)
plt.xlabel("Episode")
plt.ylabel("JFI")
plt.show()
