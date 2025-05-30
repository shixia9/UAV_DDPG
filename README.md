> source:
>
> Computation offloading optimization for UAV-assisted mobile edge computing: a deep deterministic policy gradient approach

| environment   | version   |
| ---------- | ------ |
| python     | 3.7.16 |
| tensorflow | 1.14.0 |
| gym        | 0.15.3 |

**File structure**

UAV_DDPG
├─ TD3  *//TD3算法*
│  ├─ state_normalization.py  *//状态归一化*
│  ├─ TD3_algo.py  *//训练主函数*
│  └─ UAV_env.py  *//训练环境*
├─ Local_only  *//本地计算*
│  └─ Local_only.py
├─ Edge_only  *//全部卸载计算*
│  └─ Edge_only.py
├─ DQN  *//DQN算法*
│  ├─ dqn_algo.py
│  ├─ state_normalization.py
│  └─ UAV_env.py
├─ DDPG  *//DDPG算法*
│  ├─ ddpg_algo.py
│  ├─ state_normalization.py
│  ├─ UAV_env.py
│  ├─ DDPG_without_state_normalization  *//未经过归一化的DDPG*
│  │  ├─ ddpg_algo.py
│  │  └─ UAV_env.py
│  └─ DDPG_without_behavior_noise  *//未加入噪声的DDPG*
│     ├─ ddpg_algo.py
│     ├─ state_normalization.py
│     └─ UAV_env.py
└─ Actor Critc  *//演员-评论家算法*
   ├─ ac_algo.py
   ├─ state_normalization.py
   └─ UAV_env.py



**Run**

+ TD3: `TD3_algo.py`
+ Local only: `Local_only.py`
+ Edge only: `Edge_only.py`
+ DQN: `dqn_algo.py`
+ DDPG: `ddpg_algo.py`
+ Actor Critic: `ac_algo` 
