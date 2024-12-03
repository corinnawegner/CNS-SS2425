"""
For delays d ∈ [−0.1s,0.1s] and steps of dt determine the weight change of w1 after one spike. Choose a
 simulation time which allows you to approximately integrate over the whole pre- and postsynaptic traces
 and justify your choice in the protocol.
 Plot the final weight change for a single spike depending on the delay d (Note: Do not plot the time
 course of the weights). Add this plot to the protocol.
In your protocol, compare this curve to the experimentally found biological weight changes due to spike
timing-dependent plasticity (see lecture, Bi and Poo, J Neuroscience, 1998). Also, discuss between which
 signals the time differences were measured. Are the weight change curves comparable?
"""

from learning_rules import Neuron, Synapse
import numpy as np
import matplotlib.pyplot as plt

dt = 0.001
learning_rate = 0.1
tmax = 2

list_d = np.linspace(-0.1,0.1, 20)
list_weights = []

for d in list_d:

    neuron_1 = Neuron(5, total_spiking_time = tmax)
    neuron_2 = Neuron(5, total_spiking_time = tmax, delay = d)
    post_neuron = Neuron(0)
    synapse_1 = Synapse(neuron_1, post_neuron, learning_rate)
    synapse_2 = Synapse(neuron_2, post_neuron, learning_rate, weight=1)

    neuron_1.generate_spike_train(dt)
    neuron_2.generate_spike_train(dt)

    list_t = np.linspace(0, tmax, int(tmax/dt))
    list_weight = []

    for t in list_t:
        list_weight.append(synapse_1.weight)
        post_neuron.generate_output_spike_train([synapse_1.weight, synapse_2.weight], [neuron_1.convolved_spike_train, neuron_2.convolved_spike_train])
        synapse_1.update_weight(learning_rule='iso', delta_t=dt, t=t)

    weight_final = synapse_1.weight
    list_weights.append(weight_final)

plt.plot(list_d, list_weights)
plt.title("Final weight change curve")
plt.xlabel("Delay d")
plt.ylabel("Weight")
plt.show()
