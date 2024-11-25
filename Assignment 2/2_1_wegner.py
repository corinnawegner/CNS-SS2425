from learning_rules import Neuron, Synapse, double_exponential
import numpy as np
import matplotlib.pyplot as plt

time_de = np.linspace(0,2, 1000)
list_de = [double_exponential(t) for t in time_de]
plt.plot(time_de, list_de)
plt.xlabel('Time')
plt.ylabel('double exponential')
plt.title("Double exponential function")
plt.show()

dt = 0.001
learning_rate = 0.01
tmax = 10

neuron_1 = Neuron(5, total_spiking_time = tmax)
neuron_2 = Neuron(5, total_spiking_time = 4, delay = -0.004) # not learning, Spike train with d = -40 ms
post_neuron = Neuron(0)
synapse_1 = Synapse(neuron_1, post_neuron, learning_rate)
synapse_2 = Synapse(neuron_2, post_neuron, learning_rate, weight=1)

neuron_2.generate_spike_train(dt)

list_t = np.linspace(0, tmax, int(tmax/dt))
list_weight = []

for t in list_t:
    list_weight.append(synapse_1.weight)
    post_neuron.update_rate([synapse_1.weight, synapse_2.weight], [neuron_1.rate, neuron_2.rate])
    synapse_1.update_weight(learning_rule='iso', delta_t=dt, t=t)

plt.plot(list_t, list_weight)
plt.title("Weight change for Synapse 1 with delay -0.004")
plt.xlabel("Time")
plt.ylabel("Weight")
plt.show()

neuron_3 = Neuron(5, total_spiking_time = tmax)
neuron_4 = Neuron(5, total_spiking_time = 4, delay = 0.004) # not learning, Spike train with d = -40 ms
post_neuron_2 = Neuron(0)
synapse_3 = Synapse(neuron_3, post_neuron_2, learning_rate)
synapse_4 = Synapse(neuron_4, post_neuron_2, learning_rate, weight=1)

neuron_4.generate_spike_train(dt)

list_weight = []

for t in list_t:
    list_weight.append(synapse_3.weight)
    post_neuron_2.update_rate([synapse_3.weight, synapse_4.weight], [neuron_3.rate, neuron_4.rate])
    synapse_3.update_weight(learning_rule='iso', delta_t=dt, t=t)

plt.plot(list_t, list_weight)
plt.title("Weight change for Synapse 1 with delay 0.004")
plt.xlabel("Time")
plt.ylabel("Weight")
plt.show()