from learning_rules import Neuron, Synapse
import numpy as np
import matplotlib.pyplot as plt

#Setting the constants
dt = 0.001
learning_rate = 0.1
tmax = 10
list_t = np.linspace(0, tmax, int(tmax / dt))

for delay in [-0.04, 0.04]:
    # Define the system
    neuron_1 = Neuron(5, total_spiking_time = tmax)
    neuron_2 = Neuron(5, total_spiking_time = tmax, delay = delay, stop_after=4)
    post_neuron = Neuron(0)
    synapse_1 = Synapse(neuron_1, post_neuron, learning_rate)
    synapse_2 = Synapse(neuron_2, post_neuron, learning_rate, weight=1)

    neuron_1.generate_spike_train(dt)
    neuron_2.generate_spike_train(dt)

    # Simulate the weigth change

    list_weight = []

    for t in list_t:
        list_weight.append(synapse_1.weight)
        post_neuron.generate_output_spike_train([synapse_1.weight, synapse_2.weight],
                                                [neuron_1.convolved_spike_train, neuron_2.convolved_spike_train])
        synapse_1.update_weight(learning_rule='ico', delta_t=dt, t=t)

    plt.plot(list_t, list_weight)
    plt.title(f"Weight change for Synapse 1 with delay {delay} using ICO rule")
    plt.xlabel("Time")
    plt.ylabel("Weight")
    plt.show()


