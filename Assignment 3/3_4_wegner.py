import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm
from RL_environment import *

def exponential_decay(n, A, B, lam):
    return A*np.exp(-n/lam) + B

RL_env = create_environment(21, 21)

list_epsilon = [0, 0.2, 0.4, 0.6, 0.8]
list_avg_paths = []
list_fit_params = []

n_epochs = np.arange(1000)

for eps in tqdm(list_epsilon, desc="Processing epsilon values"):
    list_path_length = []
    for i in range(100):
        list_path_length_i = []
        for epoch in n_epochs:
            RL_env.back_to_start_position()
            while tuple(RL_env.pos) != tuple(RL_env.goal_state):
                RL_env.epsilon_step(0.2)
                RL_env.update_value(RL_env.positions[-2]) #We update the state before the current step
            list_path_length_i.append(len(RL_env.positions))
        list_path_length.append(list_path_length_i)

    mean_path_length = np.mean(list_path_length, axis=0)
    popt, _ = curve_fit(exponential_decay, n_epochs, mean_path_length, p0=(1, 100, 1))

    list_avg_paths.append(mean_path_length)
    list_fit_params.append(popt)

for eps, path, popt in zip(list_epsilon, list_avg_paths, list_fit_params):
    plt.plot(n_epochs, path, label = f"Average path lengths for epsilon = {eps}")
    plt.plot(n_epochs, exponential_decay(n_epochs, *popt), label=f"Fitted Function for epsilon = {eps}")
plt.xlabel("epoch")
plt.ylabel("Path length")
plt.title("Average path lengths over 100 trials")
plt.legend()
plt.show()

plt.plot(list_epsilon, list_fit_params[:,1])
plt.title("Dependency of parameter B on epsilon")
plt.xlabel("epsilon")
plt.ylabel("fit parameter")
plt.show()

plt.plot(list_epsilon, list_fit_params[:,2])
plt.title("Dependency of parameter Lambda on epsilon")
plt.xlabel("epsilon")
plt.ylabel("fit parameter")
plt.show()