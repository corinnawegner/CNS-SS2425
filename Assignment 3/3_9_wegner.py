from scipy.optimize import curve_fit
from RL_environment import *


list_epsilon = [0, 0.2, 0.4, 0.6, 0.8]
list_avg_paths = []
list_fit_params = []

n_epochs = np.arange(1000)

eps = 0.2

RL_env = create_environment(21, 21, learning_mode = "Actor-Critic")
list_path_length = []
for i in range(100):
    list_path_length_i = []
    for epoch in n_epochs:
        RL_env.back_to_start_position()
        while tuple(RL_env.pos) != tuple(RL_env.goal_state):
            RL_env.softmax_step(eps)
            RL_env.update_value(RL_env.positions[-2]) #We update the state before the current step
        list_path_length_i.append(len(RL_env.positions))
    list_path_length.append(list_path_length_i)

mean_path_length = np.mean(list_path_length, axis=0)
popt, _ = curve_fit(exponential_decay, n_epochs, mean_path_length, p0=(1, 100, 1))

list_avg_paths.append(mean_path_length)
list_fit_params.append(popt)

plt.plot(n_epochs, list_avg_paths[0], label = f"Average path lengths for epsilon = {eps}")
plt.plot(n_epochs, exponential_decay(n_epochs, *popt), label=f"Fitted Function for epsilon = {eps}")
plt.xlabel("epoch")
plt.ylabel("Path length")
plt.title("Average path lengths over 100 trials")
plt.legend()
plt.show()