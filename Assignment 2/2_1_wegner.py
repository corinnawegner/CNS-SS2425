from learning_rules import Neuron, Synapse, double_exponential
import numpy as np
import matplotlib.pyplot as plt

#Setting the variables
dt = 0.001
learning_rate = 0.1
tmax = 10
list_t = np.linspace(0, tmax, int(tmax/dt))

#Plotting the double exponential function
time_de = np.linspace(0,0.25, 1000)
list_de = [double_exponential(t) for t in time_de]
plt.plot(time_de, list_de)
plt.xlabel('Time')
plt.ylabel('Double exponential')
plt.title("Double exponential function")
plt.show()

# Defining the system for delay d = -40 ms
neuron_1 = Neuron(5, total_spiking_time = tmax)
neuron_2 = Neuron(5, total_spiking_time = tmax, delay = -0.04, stop_after=4) # not learning, Spike train with d = -40 ms
post_neuron = Neuron(0)
synapse_1 = Synapse(neuron_1, post_neuron, learning_rate)
synapse_2 = Synapse(neuron_2, post_neuron, learning_rate, weight=1)

neuron_1.generate_spike_train(dt)
neuron_2.generate_spike_train(dt)

#Simulating the system for d = -40 ms
list_weight = []

for t in list_t:
    list_weight.append(synapse_1.weight)
    post_neuron.generate_output_spike_train([synapse_1.weight, synapse_2.weight], [neuron_1.convolved_spike_train, neuron_2.convolved_spike_train])
    synapse_1.update_weight(learning_rule='iso', delta_t=dt, t=t)

# Plotting results for d = -40 ms
plt.plot(list_t, list_weight)
plt.title("Weight change for Synapse 1 with delay -0.04 using ISO rule")
plt.xlabel("Time")
plt.ylabel("Weight")
plt.show()

#Defining the system for delay d = 40 ms
neuron_3 = Neuron(5, total_spiking_time = tmax)
neuron_4 = Neuron(5, total_spiking_time = tmax, delay = 0.04, stop_after=4) # not learning, Spike train with d = -40 ms
post_neuron_2 = Neuron(0)
synapse_3 = Synapse(neuron_3, post_neuron_2, learning_rate)
synapse_4 = Synapse(neuron_4, post_neuron_2, learning_rate, weight=1)

neuron_3.generate_spike_train(dt)
neuron_4.generate_spike_train(dt)

#Simulating the system for d = 40 ms
list_weight = []

for t in list_t:
    list_weight.append(synapse_3.weight)
    #post_neuron_2.update_rate([synapse_3.weight, synapse_4.weight], [neuron_3.rate, neuron_4.rate])
    post_neuron_2.generate_output_spike_train([synapse_3.weight, synapse_4.weight], [neuron_3.convolved_spike_train, neuron_4.convolved_spike_train])
    synapse_3.update_weight(learning_rule='iso', delta_t=dt, t=t)

# Plotting results for d = 40 ms
plt.plot(list_t, list_weight)
plt.title("Weight change for Synapse 1 with delay 0.04 using ISO rule")
plt.xlabel("Time")
plt.ylabel("Weight")
plt.show()

