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
from utils.rewards import reward_basic

# Pathes
nowtime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
epoch_cnt = int(sys.argv[2])
first_rain_event_num = int(sys.argv[3])  # 1
final_rain_event_num = int(sys.argv[4])  # 5
low_depth_penalty = eval(sys.argv[5])
result_folder_name = f"{sys.argv[1]}_{nowtime}_rain_{first_rain_event_num}_{final_rain_event_num}_episode_{epoch_cnt}_low_penalty_{low_depth_penalty}"

training_cases_path = r"../../data/"
training_cases_name = r"rain_case"
results_path = r"../../results/training/" + result_folder_name + "/"
if not os.path.exists(results_path):
    os.makedirs(results_path)
SWMM_outputs_path = results_path + r"SWMM_outputs/"
if not os.path.exists(SWMM_outputs_path):
    os.makedirs(SWMM_outputs_path)
csv_save_path = results_path + r"rewards_csv/"
if not os.path.exists(csv_save_path):
    os.makedirs(csv_save_path)
trained_model_path = results_path + r"trained_models/"
if not os.path.exists(trained_model_path):
    os.makedirs(trained_model_path)
ckpt_path = trained_model_path + r"ckpt/"
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
save_model_freq = 2000

# reward function def
reward_function = reward_basic
target_depths = [3.4, 4.7]
depth_penalty = [-30, -100]
depth_advantage = [0, 0]

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
steps_per_epoch = 1 * rain_duration
gamma = 0.99

# Buffer
buffer = Buffer(
    observation_dimensions, buffer_size=steps_per_epoch, gamma=gamma
)

# Agent
pg_agent = PolicyGradientAgent(observation_dimensions, action_dimensions, buffer)
starttime = datetime.datetime.now()

# Train
mean_return_per_epoch = {}
for epoch in range(epoch_cnt):
    sum_return = 0
    num_episode = 0
    episode_return = 0
    # sample a rain event
    current_rain_event_num = random.randint(first_rain_event_num, final_rain_event_num)
    rain = rain_generation(current_rain_event_num)
    # pySWMM
    swmm_model = PySWMM(
        f"{training_cases_path}{training_cases_name}{current_rain_event_num}.inp",
        f"{SWMM_outputs_path}{training_cases_name}{current_rain_event_num}.rpt",
        f"{SWMM_outputs_path}{training_cases_name}{current_rain_event_num}.out",
    )
    swmm_model.swmm_open()
    swmm_model.swmm_start()

    print(f"Epoch: {epoch + 1} / {epoch_cnt}, rain event: {current_rain_event_num}")
    terminal = False
    last_step_action = 0
    num_opened_pumps_1 = 0
    num_opened_pumps_2 = 0

    # Init state
    obs_state = np.zeros(observation_dimensions)
    obs_state[0] = swmm_model.getNodeResults(node[0], 5)
    obs_state[1] = swmm_model.getNodeResults(node[1], 5)
    # Rain prediction in 30 mins
    for i in range(2, 8):
        obs_state[i] = rain[step + i - 2]
    obs_state[8] = num_opened_pumps_1
    obs_state[9] = num_opened_pumps_2

    for step in range(steps_per_epoch):
        # Take action
        if step > 0:
            last_step_action = action
        action = pg_agent.sample_action(obs_state)

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
        obs_state_new = np.zeros(observation_dimensions)
        obs_state_new[0] = swmm_model.getNodeResults(node[0], 5)
        obs_state_new[1] = swmm_model.getNodeResults(node[1], 5)
        for i in range(2, 8):
            obs_state_new[i] = rain[step % rain_duration + i - 2]
        obs_state_new[8] = num_opened_pumps_1
        obs_state_new[9] = num_opened_pumps_2

        # reward
        reward = reward_function(
            [obs_state_new[0], obs_state_new[1]],
            target_depths=target_depths,
            depth_penalty=depth_penalty,
            depth_advantage=depth_advantage,
        )
        episode_return += reward

        # store obs, act, reward
        pg_agent.buffer.store(obs_state, action, reward)

        # state transition
        obs_state = obs_state_new

        # finish one rollout
        if terminal or step == steps_per_epoch - 1:
            pg_agent.buffer.finish_trajectory()
            num_episode += 1
            sum_return += episode_return
            episode_return = 0
            # close SWMM
            swmm_model.swmm_end()
            swmm_model.swmm_report()
            swmm_model.swmm_close()
            # reopen SWMM
            if step < steps_per_epoch - 1:
                # pySWMM
                swmm_model = PySWMM(
                    f"{training_cases_path}{training_cases_name}{current_rain_event_num}.inp",
                    f"{SWMM_outputs_path}{training_cases_name}{current_rain_event_num}.rpt",
                    f"{SWMM_outputs_path}{training_cases_name}{current_rain_event_num}.out",
                )
                swmm_model.swmm_open()
                swmm_model.swmm_start()

    # get samples from buffer
    (
        observation_buffer,
        action_buffer,
        return_buffer,
    ) = pg_agent.buffer.get()

    # Train
    pg_agent.train_policy(observation_buffer, action_buffer, return_buffer)

    print(f"Epoch: {epoch + 1}, mean return: {sum_return / num_episode}")
    if current_rain_event_num in mean_return_per_epoch:
        mean_return_per_epoch[current_rain_event_num].append(np.mean(sum_return / num_episode))
    else:
        mean_return_per_epoch[current_rain_event_num] = [np.mean(sum_return / num_episode)]

# save trained models
pg_agent.save_policy_weights(ckpt_path + "node_" + node_controlled[0] + "_policy_final.h5")

endtime = datetime.datetime.now()
print("elapsed time:", endtime - starttime)

# reward
for i in range(first_rain_event_num, final_rain_event_num + 1):
    np.savetxt(csv_save_path        
        + "training_rewards_rain_"
        + str(i)
        + "_ending_episode_"
        + str(epoch_cnt)
        + ".csv",
        mean_return_per_epoch[i],
        delimiter=","
        )
    plt.plot(mean_return_per_epoch[i], label="rain event " + str(i))

plt.legend()
plt.xlabel("Episode")
plt.ylabel("Mean reward")
plt.savefig(results_path + "ending_episode_" + str(epoch_cnt) + "_fig0.png")


