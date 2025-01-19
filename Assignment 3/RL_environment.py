import numpy as np
import matplotlib.pyplot as plt

class RL_environment():
    def __init__(self, width, height, goal_state, reward_function, pos_start=np.array([0, 0]), learning_rate = 0.1, discount_rate = 0.7,  learning_mode = "Q-Learning", beta = 0.1, gamma = 0.7):
        self.width = width
        self.height = height
        self.goal_state = tuple(goal_state)
        self.reward_function = reward_function
        self.pos_start = tuple(pos_start)
        self.pos = np.copy(pos_start)
        self.positions = [self.pos]  # Store as tuple for immutability
        self.action_hist = []
        self.values = np.zeros((width, height, 5))
        self.mode = learning_mode
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        #For Actor-Critic:
        self.gamma = gamma
        self.beta = beta

    def back_to_start_position(self):
        self.pos = self.pos_start
        self.positions = [self.pos]

    def compute_new_position(self, action):
        x, y = self.pos
        if action == "u":
            y = (y - 1) % self.height  # Wrap around
        elif action == "d":
            y = (y + 1) % self.height
        elif action == "r":
            x = (x + 1) % self.width
        elif action == "l":
            x = (x - 1) % self.width
        return (x, y)

    def action_to_step(self, action):
        pos_new = self.compute_new_position(action)
        self.pos = pos_new
        self.positions.append(self.pos)

    def do_action(self, action):
        self.action_to_step(action)
        self.action_hist.append(action)

    def determine_epsilon_action(self, epsilon):
        actions = ["u", "d", "r", "l"]
        random_number = np.random.random()

        if random_number < epsilon:
            step = np.random.choice(actions)
        else:
            current_state_values = self.values[self.pos[0], self.pos[1], :4]
            max_value_indices = np.where(current_state_values == np.max(current_state_values))[0]
            step = np.random.choice([actions[i] for i in max_value_indices])
        return step

    def epsilon_step(self, epsilon):
        step = self.determine_epsilon_action(epsilon)
        self.do_action(step)

    def action_to_index(self, action):
        action_to_index_dict = {"u": 0, "d": 1, "r": 2, "l": 3}
        return action_to_index_dict[action]

    def softmax_step(self, epsilon, wait = False):
        if epsilon == 1 or epsilon == 0:
            self.epsilon_step(epsilon, wait=wait)
            return 0

        actions = ["u", "d", "r", "l"]
        random_number = np.random.random()

        if random_number < epsilon:
            action = np.random.choice(actions)

        else:
            t = temperature(epsilon)

            #action_to_index = {"u": 0, "d": 1, "r": 2, "l": 3}
            weights = {}
            normalization_constant = 0
            for action in actions:
                q = self.values[self.pos[0], self.pos[1], self.action_to_index(action)]
                #print(f'q, t: ', q, t, q/t)
                weight_factor = np.exp(q/t)
                if weight_factor < 1e-5:
                    weights[action] = 0
                if not np.isnan(weight_factor) and not np.isinf(weight_factor):
                    weights[action] = weight_factor
                else:
                    weights[action] = 1

            # Normalize weights to probabilities
            probabilities = {action: weight / np.sum(list(weights.values())) for action, weight in weights.items()}
            #print(weights, probabilities)
            #action = actions[np.argmax(probabilities)]

            # Select an action randomly with the computed probabilities
            actions_list = list(probabilities.keys())
            probabilities_list = list(probabilities.values())
            action = np.random.choice(actions_list, p=probabilities_list)

        if not wait:
            self.action_to_step(action)  # immediately move

        self.action_hist.append(action)

    def update_value(self, position, next_action = None):
        a = self.action_hist[-1]
        z = self.action_to_index(a)
        if self.mode == "Q-Learning":
            max_q = np.max(self.values[self.pos[0], self.pos[1]][:4])
            update_factor = self.reward_function[self.pos[0], self.pos[1]] + self.discount_rate * max_q - self.values[position[0], position[1], z]
            self.values[position[0], position[1], z] += self.learning_rate * update_factor
        elif self.mode == "SARSA":
            q_val_next = self.values[self.pos[0], self.pos[1], self.action_to_index(next_action)]
            update_factor = self.reward_function[self.pos[0], self.pos[1]] + self.discount_rate * q_val_next - self.values[position[0], position[1], z]
            self.values[position[0], position[1], z] += self.learning_rate * update_factor
        elif self.mode == "Actor-Critic":
            state_prev = position
            delta_error = self.reward_function[self.pos[0], self.pos[1]] + self.gamma * self.values[self.pos[0], self.pos[1], 4] - self.values[state_prev[0], state_prev[1], 4]
            self.values[position[0], position[1], z] += self.learning_rate * delta_error
            self.values[state_prev[0], state_prev[1], 4] += self.beta * delta_error
        else:
            raise ValueError("Invalid learning mode. Choose between Q-Learning, SARSA and Actor-Critic, where default is Q-Learning.")

def create_environment(width, height, learning_mode = "Q-Learning"):
    goal_state = int(width/2), int(height/2)
    rf = np.zeros((width, height))
    rf[goal_state[0], goal_state[1]] = 100
    env = RL_environment(width, height, goal_state, rf, learning_mode = learning_mode)
    return env

def plot_results(env, title = None):
    # Extract Q_max(s) values
    Q_max = np.max(env.values, axis=2)

    # Plot Q_max(s)
    plt.figure(figsize=(12, 8))
    plt.imshow(Q_max.T, origin='lower', cmap='viridis', extent=[0, env.width, 0, env.height])
    plt.colorbar(label="Maximal Q-Value")

    # Overlay the final path
    positions = np.array(env.positions)
    plt.plot(positions[:, 0] + 0.5, positions[:, 1] + 0.5, color='red', marker='o', label='Final Path')

    # Mark the start and goal states
    plt.scatter(env.pos_start[0] + 0.5, env.pos_start[1] + 0.5, color='blue', label='Start', zorder=5)
    plt.scatter(env.goal_state[0] + 0.5, env.goal_state[1] + 0.5, color='green', label='Goal', zorder=5)

    # Add grid and labels
    plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
    plt.xticks(range(env.width))
    plt.yticks(range(env.height))
    plt.title("Maximal State-Action Values and Final Path")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.legend()
    plt.show()

def exponential_decay(n, A, B, lam):
    return A*np.exp(-n/lam) + B

def temperature(eps):
    T = eps / (1 - eps)
    return T