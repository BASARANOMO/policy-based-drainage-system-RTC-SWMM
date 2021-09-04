import numpy as np
import sys

sys.path.append("..")

import os
import datetime
import random
from pyswmm.swmm5 import PySWMM
import matplotlib.pyplot as plt

from REINFORCE.PG import PolicyGradientAgent
from utils.memory_buffer import Buffer
from utils.rain_generator import rain_generation
from utils.rewards import reward_3obj_lowdepthpenalty as reward_function

# Model
model_path = (
    f"../../results/training/test_pg_244_2021-09-04-21-27_rain_1_5_episode_3000_low_depth_penalty_True/"
    + "trained_models/ckpt/node_ChengXi_policy_final.h5"
)

# Pathes
nowtime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
low_depth_penalty = eval(sys.argv[3])
result_folder_name = f"{sys.argv[2]}_low_depth_penalty_{low_depth_penalty}"

training_cases_path = r"../../data/"
training_cases_name = r"rain_case"
results_path = r"../../results/testing/" + result_folder_name + "/"
if not os.path.exists(results_path):
    os.makedirs(results_path)
SWMM_outputs_path = results_path + r"SWMM_outputs/"
if not os.path.exists(SWMM_outputs_path):
    os.makedirs(SWMM_outputs_path)
reward_file_path = results_path + "reward.txt"


# data
rain_event = sys.argv[1]
save_depth_fig_path = (
    r"../../results/testing/" + result_folder_name
    + "/depth_rain_event_"
    + str(rain_event)
    + ".png"
)
save_pumps_fig_path = (
    r"../../results/testing/" + result_folder_name
    + "/pumps_rain_event_"
    + str(rain_event)
    + ".png"
)
show_plt_flag = False

# reward function def
target_depths = [3.4, 4.7]
low_target_depths = [1, 1]
depth_penalty = [-30, -100]
if low_depth_penalty:
    low_depth_penalty = [30, 100]
else:
    low_depth_penalty = [0, 0]
depth_advantage = [20, 20]
energy_coeff = [-100]
safety_coeff = [-10]
k_coeff1 = [0.8, 0.1, 0.1]
k_coeff2 = [1.0, 1.0, 1.0]
k_coeff2 = [x / 3 for x in k_coeff2]

# SWMM params
rain_duration = 72  # 6hours
SWMM_stride = 300  # 5 minutes, should correspond to the stride time of rain event data
node_controlled = ["ChengXi"]
n1 = "Tank_LaoDongLu"
n2 = "Tank_ChengXi"
p1 = "Pump_LaoDongLu1"
p2 = "Pump_LaoDongLu2"
p3 = "Pump_LaoDongLu3"
p4 = "Pump_ChengXi1"
p5 = "Pump_ChengXi2"
p6 = "Pump_ChengXi3"
p7 = "Pump_ChengXi4"
node = [n1, n2]
pump = [p1, p2, p3, p4, p5, p6, p7]

# Action
action_space = np.linspace(0, 19, 20, dtype="int")
action_dimensions = action_space.shape[0]
action_list = [[] for i in range(20)]
action_part1 = [[] for i in range(4)]
action_part2 = [[] for i in range(5)]
action_part1[0] = [0, 0, 0]
action_part1[1] = [1, 0, 0]
action_part1[2] = [1, 1, 0]
action_part1[3] = [1, 1, 1]
action_part2[0] = [0, 0, 0, 0]
action_part2[1] = [1, 0, 0, 0]
action_part2[2] = [1, 1, 0, 0]
action_part2[3] = [1, 1, 1, 0]
action_part2[4] = [1, 1, 1, 1]
for i in range(4):
    for j in range(5):
        action_list[5 * i + j] = action_part1[i] + action_part2[j]

# REINFORCE params
observation_dimensions = 10
gamma = 0.99

# Buffer
buffer = Buffer(
    observation_dimensions, buffer_size=rain_duration, gamma=gamma
)

# Agent
pg_agent = PolicyGradientAgent(observation_dimensions, action_dimensions, buffer)
pg_agent.set_policy_weights(model_path)
starttime = datetime.datetime.now()

# Test
## Note down results
rewards = []
rewards_1 = []
rewards_2 = []
rewards_3 = []
depths_1 = []
depths_2 = []
num_opened_pumps_list_1 = []
num_opened_pumps_list_2 = []

# pySWMM
rain = rain_generation(int(rain_event))
swmm_model = PySWMM(
    f"{training_cases_path}{training_cases_name}{rain_event}.inp",
    f"{SWMM_outputs_path}{training_cases_name}{rain_event}.rpt",
    f"{SWMM_outputs_path}{training_cases_name}{rain_event}.out",
)
swmm_model.swmm_open()
swmm_model.swmm_start()

terminal = False
last_step_action = 0
num_opened_pumps_1 = 0
num_opened_pumps_2 = 0

# Init state
obs_state = np.zeros((1, observation_dimensions))
obs_state[0][0] = swmm_model.getNodeResult(node[0], 5)
obs_state[0][1] = swmm_model.getNodeResult(node[1], 5)
# Rain prediction in 30 mins
for i in range(2, 8):
    obs_state[0][i] = rain[i - 2]
obs_state[0][8] = num_opened_pumps_1
obs_state[0][9] = num_opened_pumps_2

for step in range(rain_duration):
    # Take action
    if step > 0:
        last_step_action = action
    logits, action = pg_agent.sample_action(obs_state)
    action = action[0]

    num_opened_pumps_1 = 0
    num_opened_pumps_2 = 0
    for j in range(7):
        if j < 3:
            num_opened_pumps_1 += action_list[action][j]
        else:
            num_opened_pumps_2 += action_list[action][j]
        swmm_model.setLinkSetting(pump[j], action_list[action][j])
    
    # run swmm step
    time = swmm_model.swmm_stride(SWMM_stride)
    if time <= 0.0:
        terminal = True

    # observe the new state
    obs_state_new = np.zeros((1, observation_dimensions))
    obs_state_new[0][0] = swmm_model.getNodeResult(node[0], 5)
    obs_state_new[0][1] = swmm_model.getNodeResult(node[1], 5)
    for i in range(2, 8):
        obs_state_new[0][i] = rain[step % rain_duration + i - 2]
    obs_state_new[0][8] = num_opened_pumps_1
    obs_state_new[0][9] = num_opened_pumps_2

    # reward
    if (
        obs_state_new[0][0] >= target_depths[0]
        or obs_state_new[0][1] >= target_depths[1]
        or np.sum(obs_state_new[0][2:8]) >= 20
    ):
        k_coeff = k_coeff1
    else:
        k_coeff = k_coeff2
    reward, reward_depth, reward_energy, reward_safety = reward_function(
        [obs_state_new[0][0], obs_state_new[0][1]],
        action_list,
        action,
        last_step_action,
        target_depths=target_depths,
        low_target_depths=low_target_depths,
        depth_penalty=depth_penalty,
        low_depth_penalty=low_depth_penalty,
        depth_advantage=depth_advantage,
        energy_coeff=energy_coeff,
        safety_coeff=safety_coeff,
        k_coeff=k_coeff,
    )
    rewards.append(reward)
    rewards_1.append(reward_depth)
    rewards_2.append(reward_energy)
    rewards_3.append(reward_safety)
    depths_1.append(obs_state_new[0][0])
    depths_2.append(obs_state_new[0][1])
    num_opened_pumps_list_1.append(num_opened_pumps_1)
    num_opened_pumps_list_2.append(num_opened_pumps_2)

    # state transition
    obs_state = obs_state_new
    last_step_action = action

    # finish one rollout
    if terminal or step == rain_duration - 1:
        pg_agent.buffer.finish_trajectory()
        # close SWMM
        swmm_model.swmm_end()
        swmm_model.swmm_report()
        swmm_model.swmm_close()

if int(rain_event) == 1:
    f = open(reward_file_path, "w")
    f.close()
f = open(reward_file_path, "r+")
f.read()
f.write(
    str(np.mean(rewards_1))
    + " "
    + str(np.mean(rewards_2))
    + " "
    + str(np.mean(rewards_3))
    + " "
    + str(np.mean(rewards))
    + "\n"
)
f.close()

plt.figure(1)
plt.plot(depths_1, label="Depth of tank LDL", color="r")
plt.axhline(y=target_depths[0], linestyle="dashed", color="r")
plt.plot(depths_2, label="Depth of tank CX", color="b")
plt.axhline(y=target_depths[1], linestyle="dashed", color="b")
if low_target_depths[0] == low_target_depths[1]:
    plt.axhline(y=low_target_depths[0], linestyle=":", color="g")
    plt.text(
        57, low_target_depths[0], "Safety depth", bbox=dict(facecolor="w", alpha=0.5)
    )
start_time = datetime.datetime(2011, 1, 2, 0, 0)
current_time = start_time
x_ticks = []
for i in range(7):
    x_ticks.append(str(current_time.strftime("%H:%M")))
    current_time += datetime.timedelta(hours=1)
plt.xticks([i * 12 for i in range(7)], x_ticks)
extraticks = [low_target_depths[i] for i in range(len(low_target_depths))] + [
    target_depths[i] for i in range(len(target_depths))
]
plt.yticks(list(plt.yticks()[0]) + extraticks)
plt.legend()
plt.xlabel("Timestep")
plt.ylabel("Depth (m)")
plt.xlim(0, 72)
plt.title("Rain event: " + str(rain_event))
plt.savefig(save_depth_fig_path)
if show_plt_flag:
    plt.show()

plt.figure(2)
plt.plot(num_opened_pumps_list_1, label="Pump station LDL", color="r")
plt.plot(num_opened_pumps_list_2, label="Pump station CX", color="b")
plt.xticks([i * 12 for i in range(7)], x_ticks)
plt.yticks([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
plt.legend()
plt.xlabel("Timestep")
plt.ylabel("Number of opened pumps")
plt.xlim(0, 72)
plt.title("Rain event: " + str(rain_event))
plt.savefig(save_pumps_fig_path)
if show_plt_flag:
    plt.show()