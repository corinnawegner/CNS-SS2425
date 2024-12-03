import numpy as np
import matplotlib.pyplot as plt

class RL_environment():
    def __init__(self, width, height, goal_state, reward_function, pos_start=np.array([0, 0]), learning_mode = "Q-Learning"):
        self.width = width
        self.height = height
        self.goal_state = goal_state
        self.reward_function = reward_function
        self.pos_start = np.copy(pos_start)
        self.pos = np.copy(pos_start)
        self.positions = [tuple(pos_start)]  # Store as tuple for immutability
        self.values = np.zeros((width, height, 4))
        self.mode = learning_mode

    def move_left(self):
        if self.pos[0] > 0:
            self.pos[0] -= 1
        else:
            self.pos[0] = self.width - 1
        self.positions.append(tuple(self.pos))

    def move_right(self):
        if self.pos[0] < self.width - 1:
            self.pos[0] += 1
        else:
            self.pos[0] = 0
        self.positions.append(tuple(self.pos))

    def move_up(self):
        if self.pos[1] > 0:
            self.pos[1] -= 1
        else:
            self.pos[1] = self.height - 1
        self.positions.append(tuple(self.pos))

    def move_down(self):
        if self.pos[1] < self.height - 1:
            self.pos[1] += 1
        else:
            self.pos[1] = 0
        self.positions.append(tuple(self.pos))

    def back_to_start_position(self):
        self.pos = np.copy(self.pos_start)
        self.positions = [tuple(self.pos_start)]

    def action_to_step(self, action):
        if action == "u":
            self.move_up()
        elif action == "d":
            self.move_down()
        elif action == "r":
            self.move_right()
        elif action == "l":
            self.move_left()

    def epsilon_step(self, epsilon):
        actions = ["u", "d", "r", "l"]
        random_number = np.random.random()

        if random_number < epsilon:
            step = np.random.choice(actions)
        else:
            current_state_values = self.values[self.pos[0], self.pos[1]]
            max_value_indices = np.where(current_state_values == np.max(current_state_values))[0]
            step = np.random.choice([actions[i] for i in max_value_indices])

        self.action_to_step(step)

    def softmax_step(self, epsilon):
        T = epsilon/(1-epsilon)
        #Todo: implement softmax policy


    def last_action(self):
        state_diff = np.array(self.positions[-1]) - np.array(self.positions[-2])
        if state_diff[0] == -1:
            return "l"
        elif state_diff[0] == 1:
            return "r"
        elif state_diff[1] == -1:
            return "u"
        else:
            return "d"

    def update_value(self, position):
        a = self.last_action()
        if a == "u":
            z = 0
        elif a == "d":
            z = 1
        elif a == "r":
            z = 2
        elif a == "l":
            z = 3
        discount_rate = 0.7
        learning_rate = 0.1
        if self.mode == "Q-Learning":
            max_q = np.max(self.values[self.pos[0], self.pos[1]])
            update_factor = self.reward_function[self.pos[0], self.pos[1]] + discount_rate * max_q - self.values[position[0], position[1], z]
            self.values[position[0], position[1], z] += learning_rate * update_factor
        elif self.mode == "SARSA":
            blabla = 0 #Todo: How is the action of this value determined?
            update_factor = self.reward_function[self.pos[0], self.pos[1]] + discount_rate * blabla - self.values[position[0], position[1], z]
            self.values[position[0], position[1], z] += learning_rate * update_factor



def create_environment(width, height, learning_mode = "Q-Learning"):
    goal_state = int(width/2), int(height/2)
    rf = np.zeros((width, height))
    rf[goal_state[0], goal_state[1]] = 100
    env = RL_environment(width, height, goal_state, rf, learning_mode=learning_mode)
    return env


def plot_results(env):
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