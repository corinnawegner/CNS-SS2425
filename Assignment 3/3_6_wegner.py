import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm
from RL_environment import *

n_epochs = np.arange(1000)

#Create the new environment
def env_relearning():
    width = 21
    height = 21
    goal_state = int(width / 2), int(height / 2)
    rf = np.zeros((width, height))
    rf[goal_state[0], goal_state[1]] = 100
    rf[goal_state[0]-1, goal_state[1]] = -100
    rf[goal_state[0]+1, goal_state[1]] = -100
    rf[goal_state[0], goal_state[1]-1] = -100
    RL_env = RL_environment(width, height, goal_state, rf)
    return RL_env, goal_state

list_path_length = []

for i in range(100):

    RL_env, goal_state = env_relearning()
    list_path_length_i_before = []
    list_path_length_i_after = []

    for epoch in n_epochs:
        RL_env.back_to_start_position()
        while tuple(RL_env.pos) != tuple(RL_env.goal_state):
            RL_env.epsilon_step(0.2)
            RL_env.update_value(RL_env.positions[-2])
        list_path_length_i_before.append(len(RL_env.positions))

    if i == 0:
        plot_results(RL_env)

    RL_env.reward_function[goal_state[0], goal_state[1]-1] = 0
    RL_env.reward_function[goal_state[0], goal_state[1]+1] = -100

    for epoch in n_epochs:
        RL_env.back_to_start_position()
        while tuple(RL_env.pos) != tuple(RL_env.goal_state):
            RL_env.epsilon_step(0.2)
            RL_env.update_value(RL_env.positions[-2])
        list_path_length_i_after.append(len(RL_env.positions))
        if i == 0 and epoch % 200 == 0:
            plot_results(RL_env)

    if i == 0:
        plot_results(RL_env)

    list_path_length_i = list_path_length_i_before + list_path_length_i_after
    #np.concatenate(list_path_length_i_before, list_path_length_i_after, axis = 1)
    list_path_length.append(list_path_length_i)

avg_path_length = np.mean(list_path_length, axis = 0)
plt.plot(np.arange(2000), avg_path_length)
plt.xlabel("epoch")
plt.ylabel("Path length")
plt.title("Average path lengths over 100 trials")
plt.legend()
plt.show()

