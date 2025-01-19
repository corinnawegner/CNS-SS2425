from RL_environment import *

RL_env = create_environment(21, 21)

for epoch in range(1000):
    RL_env.back_to_start_position()
    while tuple(RL_env.pos) != tuple(RL_env.goal_state):
        RL_env.epsilon_step(0.2)
        RL_env.update_value(RL_env.positions[-2]) #We update the state before the current step
    RL_env.update_value(RL_env.positions[-1]) #Update the value of the goal state

plot_results(RL_env)