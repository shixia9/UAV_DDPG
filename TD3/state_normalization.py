import numpy as np
from UAV_env import UAVEnv

# 将不同量纲的状态变量（如电池容量、坐标、任务大小）缩放到0-1范围内，避免因数值差异过大导致模型训练不稳定
class StateNormalization(object):
    def __init__(self):
        env = UAVEnv()
        M = env.M
        # 无人机的电池容量最大值、地面长度、地面宽度、剩余任务总量最大值（100MB）
        self.high_state = np.array(
            [5e5, env.ground_length, env.ground_width, 100 * 1048576])
        self.high_state = np.append(self.high_state, np.ones(M * 2) * env.ground_length)
        self.high_state = np.append(self.high_state, np.ones(M) * 2621440)
        self.high_state = np.append(self.high_state, np.ones(M))

        # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
        # self.high_state = np.array(
        #     [500000, 100, 100, 60 * 1048576, 100, 100, 100, 100, 100, 100, 100, 100, 2097152, 2097152, 2097152, 2097152,
        #      1, 1, 1, 1])  # uav loc, ue loc, task size, block_flag
        self.low_state = np.zeros(4 * M + 4)  # uav loc, ue loc, task size, block_flag

    def state_normal(self, state):
        return state / (self.high_state - self.low_state)