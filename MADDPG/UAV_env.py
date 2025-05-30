import math
import random

import numpy as np


class UAVEnv(object):
    height = ground_length = ground_width = 100  # 场地长宽均为100m，UAV飞行高度也是
    sum_task_size = 100 * 1048576  # 总计算任务60 Mbits --> 60 80 100 120 140
    loc_uav = [50, 50]
    bandwidth_nums = 1
    B = bandwidth_nums * 10 ** 6  # 带宽1MHz
    p_noisy_los = 10 ** (-13)  # 噪声功率-100dBm
    p_noisy_nlos = 10 ** (-11)  # 噪声功率-80dBm
    flight_speed = 50.  # 飞行速度50m/s
    f_ue = 6e8  # UE的计算频率0.6GHz
    f_uav = 1.2e9  # UAV的计算频率1.2GHz
    r = 10 ** (-27)  # 芯片结构对cpu处理的影响因子
    s = 1000  # 单位bit处理所需cpu圈数1000
    p_uplink = 0.1  # 上行链路传输功率0.1W
    alpha0 = 1e-5  # 距离为1m时的参考信道增益-30dB = 0.001， -50dB = 1e-5
    T = 400  # 周期400s
    t_fly = 1
    t_com = 9
    delta_t = t_fly + t_com  # 1s飞行, 后9s用于悬停计算
    v_ue = 1    # ue移动速度1m/s
    slot_num = int(T / delta_t)  # 40个间隔
    m_uav = 9.65  # uav质量/kg
    # ref: Mobile Edge Computing via a UAV-Mounted Cloudlet: Optimization of Bit Allocation and Path Planning
    e_battery_uav = 500000  # uav电池电量: 500kJ. 

    #################### ues ####################
    M = 4  # UE数量
    block_flag_list = np.random.randint(0, 2, M)  # 4个ue，ue的遮挡情况
    loc_ue_list = np.random.randint(0, 101, size=[M, 2])  # 位置信息:x在0-100随机
    # task_list = np.random.randint(1572864, 2097153, M)      # 随机计算任务1.5~2Mbits ->对应总任务大小60
    #task_list = np.random.randint(2097153, 2621440, M)  # 随机计算任务2~2.5Mbits -> 80
    task_list = np.random.randint(2621440, 3145729, M)  # 随机计算任务2.5~3Mbits -> 100

    action_bound = [-1, 1]  # 对应tahn激活函数
    action_dim = 4  # 第一位表示服务的ue id;中间两位表示飞行角度和距离；后1位表示目前服务于UE的卸载率
    state_dim = 4 + M * 4  # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag

    #################### uav ####################
    num_agents = 2  # UAV数量

    def __init__(self):
        # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
        # 初始化所有无人机状态
        self.uavs = [{
            'loc': [50 + 10*i, 50],  # 初始位置分散
            'battery': self.e_battery_uav,
        } for i in range(self.num_agents)]
        # UE相关初始化
        self.block_flag_list = np.random.randint(0, 2, self.M)
        self.loc_ue_list = np.random.randint(0, 101, size=[self.M, 2])
        self.task_list = np.random.randint(2621440, 3145729, self.M)
        # XXX: 新扩展后的状态空间
        # N * (uav battery remain + uav loc) + M * (ue loc + ue task size + ue block_flag) + remaining sum task size
        self.state_dim = 3 * self.num_agents + 4 * self.M + 1
        self.action_dim = 4

    def reset_env(self):
        self.sum_task_size = 100 * 1048576  # 总计算任务60 Mbits -> 60 80 100 120 140
        # self.e_battery_uav = 500000  # uav电池电量: 500kJ
        # self.loc_uav = [50, 50]
        for i in range(self.num_agents):
            self.uavs[i]['loc'] = [50 + 10*i, 50]
            self.uavs[i]['battery'] = self.e_battery_uav
        self.loc_ue_list = np.random.randint(0, 101, size=[self.M, 2])  # 位置信息:x在0-100随机
        self.reset_step()

    def reset_step(self):
        self.task_list = np.random.randint(2621440, 3145729, self.M)  # 随机计算任务1.5~2Mbits -> 1.5~2 2~2.5 *2.5~3 3~3.5 3.5~4
        self.block_flag_list = np.random.randint(0, 2, self.M)  # 4个ue，ue的遮挡情况

    def reset(self):
        self.reset_env()
        # # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
        # self.state = np.append(self.e_battery_uav, self.loc_uav)
        # self.state = np.append(self.state, self.sum_task_size)
        # self.state = np.append(self.state, np.ravel(self.loc_ue_list))
        # self.state = np.append(self.state, self.task_list)
        # self.state = np.append(self.state, self.block_flag_list)
        return self._get_obs()

    def _get_obs(self):
        # # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
        # self.state = np.append(self.e_battery_uav, self.loc_uav)
        # self.state = np.append(self.state, self.sum_task_size)
        # self.state = np.append(self.state, np.ravel(self.loc_ue_list))
        # self.state = np.append(self.state, self.task_list)
        # self.state = np.append(self.state, self.block_flag_list)
        # return self.state
        state = []
        for uav in self.uavs:
            state.extend([
                uav['battery'],
                uav['loc'][0],
                uav['loc'][1]
            ])
        state.extend(np.ravel(self.loc_ue_list))
        state.extend(self.task_list)
        state.extend(self.block_flag_list)
        return np.array(state)

    # 对于action[4]--------0: 选择服务的ue编号 ; 1: 方向theta; 2: 距离d; 3: offloading ratio
    def step(self, actions):
        
        rewards = []
        step_redo_flag = [False] * self.num_agents
        is_terminal = False
        infos = {'offloading_changes': [], 'reset_dists': []}
        # 并行处理每个无人机的动作
        for agent_id in range(self.num_agents):
            action = actions[agent_id]
            uav = self.uavs[agent_id]
            original_loc = uav['loc'].copy()
            # 动作解码（同单智能体逻辑）
            action = (action + 1) / 2  # [-1,1] -> [0,1]
            if action[0] == 1:
                ue_id = self.M - 1
            else:
                ue_id = int(self.M * action[0])
            theta = action[1] * np.pi * 2
            offloading_ratio = action[3]
            task_size = self.task_list[ue_id]
            block_flag = self.block_flag_list[ue_id]
            # ------ 飞行与能耗计算------
            dis_fly = action[2] * self.flight_speed * self.t_fly
            dx_uav = dis_fly * math.cos(theta)
            dy_uav = dis_fly * math.sin(theta)
            new_x = uav['loc'][0] + dx_uav
            new_y = uav['loc'][1] + dy_uav
            # 冲突检测：与其他无人机距离需>5m
            collision = False
            for other_id in range(self.num_agents):
                if other_id == agent_id:
                    continue
                dist = np.sqrt((new_x - self.uavs[other_id]['loc'][0])**2 + 
                              (new_y - self.uavs[other_id]['loc'][1])**2)
                if dist < 5:
                    collision = True
                    break
            if collision:  # 发生冲突，不更新位置
                new_x, new_y = original_loc
                infos['reset_dists'].append(True)
            elif new_x < 0 or new_x > self.ground_width or new_y < 0 or new_y > self.ground_length:  # uav位置不对
                infos['reset_dists'].append(True)
            else:
                uav['loc'][0] = np.clip(new_x, 0, self.ground_width)
                uav['loc'][1] = np.clip(new_y, 0, self.ground_length)
                infos['reset_dists'].append(False)
            # 计算能耗（单个UAV）
            e_fly = (dis_fly / self.t_fly)**2 * self.m_uav * self.t_fly * 0.5
            t_server = offloading_ratio * task_size / (self.f_uav / self.s)
            e_server = self.r * self.f_uav**3 * t_server
            # ------ 任务处理与奖励计算 ------
            if self.sum_task_size == 0:
                reward = 0
                is_terminal = True
            else:
                delay = self.com_delay(self.loc_ue_list[ue_id], uav['loc'], offloading_ratio, task_size, block_flag)
                reward = -delay

            # 电量更新
            uav['battery'] -= (e_fly + e_server)
            rewards.append(reward)
        # 更新UE位置和任务（同单智能体）
        for i in range(self.M):
            theta_ue = random.random() * np.pi * 2
            dis_ue = random.random() * self.delta_t * self.v_ue
            self.loc_ue_list[i][0] = np.clip(
                self.loc_ue_list[i][0] + math.cos(theta_ue) * dis_ue, 0, self.ground_width)
            self.loc_ue_list[i][1] = np.clip(
                self.loc_ue_list[i][1] + math.sin(theta_ue) * dis_ue, 0, self.ground_length)
        self.task_list = np.random.randint(2621440, 3145729, self.M)
        if self.sum_task_size == 0:
            is_terminal = True
        return self._get_obs(), rewards, is_terminal, infos
        
        
        

    # 重置ue任务大小，剩余总任务大小，ue位置，并记录到文件
    def reset2(self, delay, x, y, offloading_ratio, task_size, ue_id):
        self.sum_task_size -= self.task_list[ue_id]  # 剩余任务量
        for i in range(self.M):  # ue随机移动后的位置
            tmp = np.random.rand(2)
            theta_ue = tmp[0] * np.pi * 2  # ue 随机移动角度
            dis_ue = tmp[1] * self.delta_t * self.v_ue  # ue 随机移动距离
            self.loc_ue_list[i][0] = self.loc_ue_list[i][0] + math.cos(theta_ue) * dis_ue
            self.loc_ue_list[i][1] = self.loc_ue_list[i][1] + math.sin(theta_ue) * dis_ue
            self.loc_ue_list[i] = np.clip(self.loc_ue_list[i], 0, self.ground_width)
        self.reset_step()  # ue随机计算任务1~2Mbits # 4个ue，ue的遮挡情况
        # 记录UE花费
        file_name = 'output.txt'
        # file_name = 'output_ddpg_' + str(self.bandwidth_nums) + 'MHz.txt'
        with open(file_name, 'a') as file_obj:
            file_obj.write("\nUE-" + '{:d}'.format(ue_id) + ", task size: " + '{:d}'.format(int(task_size)) + ", offloading ratio:" + '{:.2f}'.format(offloading_ratio))
            file_obj.write("\ndelay:" + '{:.2f}'.format(delay))
            file_obj.write("\nUAV hover loc:" + "[" + '{:.2f}'.format(x) + ', ' + '{:.2f}'.format(y) + ']')  # 输出保留两位结果


    # 计算花费
    def com_delay(self, loc_ue, loc_uav, offloading_ratio, task_size, block_flag):
        dx = loc_uav[0] - loc_ue[0]
        dy = loc_uav[1] - loc_ue[1]
        dh = self.height
        dist_uav_ue = np.sqrt(dx * dx + dy * dy + dh * dh)
        p_noise = self.p_noisy_los
        if block_flag == 1:
            p_noise = self.p_noisy_nlos
        g_uav_ue = abs(self.alpha0 / dist_uav_ue ** 2)  # 信道增益
        trans_rate = self.B * math.log2(1 + self.p_uplink * g_uav_ue / p_noise)  # 上行链路传输速率bps
        t_tr = offloading_ratio * task_size / trans_rate  # 上传时延,1B=8bit
        t_edge_com = offloading_ratio * task_size / (self.f_uav / self.s)  # 在UAV边缘服务器上计算时延
        t_local_com = (1 - offloading_ratio) * task_size / (self.f_ue / self.s)  # 本地计算时延
        if t_tr < 0 or t_edge_com < 0 or t_local_com < 0:
            raise Exception(print("+++++++++++++++++!! error !!+++++++++++++++++++++++"))
        return max([t_tr + t_edge_com, t_local_com])  # 飞行时间影响因子


