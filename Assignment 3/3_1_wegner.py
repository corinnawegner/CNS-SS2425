import numpy as np

class RL_environment():
    def __init__(self, width, height, goal_state, reward_function, pos_start = (0,0)):
        self.width = width
        self.height = height
        self.goal_state = goal_state
        self.reward_function = reward_function
        self.pos_start = pos_start
        self.pos = pos_start
        self.values = np.zeros((width, height, 4))

    def move_left(self):
        if self.pos[0] > 0:
            self.pos[0] -= 1
        else:
            self.pos[0] = self.width

    def move_right(self):
        if self.pos[0] < self.width:
            self.pos[0] += 1
        else:
            self.pos[0] = 0

    def move_up(self):
        if self.pos[1] > 0:
            self.pos[1] -= 1
        else:
            self.pos[1] = self.height

    def move_down(self):
        if self.pos[1] < self.height:
            self.pos[1] += 1
        else:
            self.pos[1] = 0

    def back_to_start_position(self):
        self.pos = self.pos_start

    def epsilon_step(self, epsilon):
        actions = ["u", "d", "r", "l"]
        random_number = np.random.random()

        if random_number < epsilon:
            step = np.random.choice(actions)
        else:
            current_state_values = self.values[self.pos[0], self.pos[1]]
            max_value_indices = np.where(current_state_values == np.max(current_state_values))[0]
            step = np.random.choice([actions[i] for i in max_value_indices])

        if step == "u":
            self.move_up()
        elif step == "d":
            self.move_down()
        elif step == "r":
            self.move_right()
        elif step == "l":
            self.move_left()

    def update_value(self, position, action):
        if action == "u":
            z = 0
        elif action == "d":
            z = 1
        elif action == "r":
            z = 2
        elif action == "l":
            z = 3
        discount_rate = 0.7
        learning_rate = 0.1
        next_state_values = self.values[(self.pos[0]+1, self.pos[1]]
        max_q = np.max(next_state_values)
        update_factor = self.reward_function[position[0], position[1]] + discount_rate * max_q - self.values[position[0], position[1], z]
        self.values[position[0], position[1], z] += learning_rate * update_factor

def create_environment(width, height):
    goal_state = int(width/2), int(height/2)
    rf = np.zeros((width, height))
    rf[goal_state[0], goal_state[1]] = 100
    env = RL_environment(width, height, goal_state, rf)
    return env