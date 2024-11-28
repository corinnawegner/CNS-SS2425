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
learning_rate = 0.01
tmax = 2

neuron_1 = Neuron(5, total_spiking_time = tmax, mode="single")
neuron_2 = Neuron(5, total_spiking_time = tmax, mode="single", delay = -0.1) # not learning, Spike train with d = -100 ms
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

plt.plot(list_t, list_weight)
plt.title("Weight change for Synapse 1 with delay -0.1")
plt.xlabel("Time")
plt.ylabel("Weight")
plt.show()

neuron_3 = Neuron(5, total_spiking_time = tmax, mode = "single")
neuron_4 = Neuron(5, total_spiking_time = tmax, mode = "single", delay = 0.1) # not learning, Spike train with d = -40 ms
post_neuron_2 = Neuron(0)
synapse_3 = Synapse(neuron_3, post_neuron_2, learning_rate)
synapse_4 = Synapse(neuron_4, post_neuron_2, learning_rate, weight=1)

neuron_3.generate_spike_train(dt)
neuron_4.generate_spike_train(dt)

list_weight = []

for t in list_t:
    list_weight.append(synapse_3.weight)
    post_neuron_2.generate_output_spike_train([synapse_3.weight, synapse_4.weight], [neuron_3.convolved_spike_train, neuron_4.convolved_spike_train])
    synapse_3.update_weight(learning_rule='iso', delta_t=dt, t=t)

plt.plot(list_t, list_weight)
plt.title("Weight change for Synapse 1 with delay 0.1")
plt.xlabel("Time")
plt.ylabel("Weight")
plt.show()

