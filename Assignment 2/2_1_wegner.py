from learning_rules import Neuron, Synapse
import numpy as np

dt = 0.001
learning_rate = 0.01

neuron_1 = Neuron(5)
neuron_2 = Neuron(5) # not learning
post_neuron = Neuron(0)
synapse_1 = Synapse(neuron_1, post_neuron, learning_rate)
synapse_2 = Synapse(neuron_2, post_neuron, learning_rate, weight=1)

neuron_1_train = neuron_1.generate_spike_train(10, 0.001)
neuron_2_train_1 = neuron_2.generate_spike_train(4, 0.001, delay=-0.004) # Spike train with d = -40 ms
neuron_2_train_2 = neuron_2.generate_spike_train(4, 0.001, delay=0.004) # Spike train with d = 40 ms

list_t = np.arange(0, 10, 0.001)
list_weight = []

for t in list_t:

    list_weight.append(synapse_1.weight)
    post_neuron.update_rate([synapse_1.weight, synapse_2.weight], [neuron_1.rate, neuron_2.rate])
    synapse_1.update_weight()