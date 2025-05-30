> 论文来源:
>
> Computation offloading optimization for UAV-assisted mobile edge computing: a deep deterministic policy gradient approach

| 运行环境   | 版本   |
| ---------- | ------ |
| python     | 3.7.16 |
| tensorflow | 1.14.0 |
| gym        | 0.15.3 |

**文件结构**

UAV_DDPG

├─ TD3
│  ├─ state_normalization.py
│  ├─ TD3_algo.py
│  └─ UAV_env.py
├─ Local_only
│  └─ Local_only.py
├─ Edge_only
│  └─ Edge_only.py
├─ DQN
│  ├─ dqn_algo.py
│  ├─ state_normalization.py
│  └─ UAV_env.py
├─ DDPG
│  ├─ ddpg_algo.py
│  ├─ state_normalization.py
│  ├─ UAV_env.py
│  ├─ DDPG_without_state_normalization
│  │  ├─ ddpg_algo.py
│  │  └─ UAV_env.py
│  └─ DDPG_without_behavior_noise
│     ├─ ddpg_algo.py
│     ├─ state_normalization.py
│     └─ UAV_env.py
└─ Actor Critc
   ├─ ac_algo.py
   ├─ state_normalization.py
   └─ UAV_env.py



**运行方式**

+ TD3: `TD3_algo.py`
+ Local only: `Local_only.py`
+ Edge only: `Edge_only.py`
+ DQN: `dqn_algo.py`
+ DDPG: `ddpg_algo.py`
+ Actor Critic: `ac_algo` 
