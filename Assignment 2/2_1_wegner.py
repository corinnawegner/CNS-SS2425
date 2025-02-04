from learning_rules import Neuron, Synapse, double_exponential
import numpy as np
import matplotlib.pyplot as plt

#Setting the constants
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

for delay in [-0.04, 0.04]:
    # Defining the system
    neuron_1 = Neuron(5, total_spiking_time = tmax)
    neuron_2 = Neuron(5, total_spiking_time = tmax, delay = delay, stop_after=4)
    post_neuron = Neuron(0)
    synapse_1 = Synapse(neuron_1, post_neuron, learning_rate)
    synapse_2 = Synapse(neuron_2, post_neuron, learning_rate, weight=1)

    neuron_1.generate_spike_train(dt)
    neuron_2.generate_spike_train(dt)

    #Simulating the weight change
    list_weight = []

    for t in list_t:
        list_weight.append(synapse_1.weight)
        post_neuron.generate_output_spike_train([synapse_1.weight, synapse_2.weight],
                                                [neuron_1.convolved_spike_train, neuron_2.convolved_spike_train])
        synapse_1.update_weight(learning_rule='iso', delta_t=dt, t=t)

    # Plotting results for d = -40 ms
    plt.plot(list_t, list_weight)
    plt.title(f"Weight change for Synapse 1 with delay {delay} using ISO rule")
    plt.xlabel("Time")
    plt.ylabel("Weight")
    plt.show()

    plt.plot(list_t[4001:], list_weight[4001:])
    plt.title(f"Weight change for Synapse 1 with delay {delay} using ISO rule, zoomed in")
    plt.xlabel("Time")
    plt.ylabel("Weight")
    plt.show()

#Varying learning rate:
list_mu = [0.05, 0.07, 0.09, 0.1, 0.2]
list_developments = []
for lr in list_mu:
    # Defining the system
    neuron_1 = Neuron(5, total_spiking_time=tmax)
    neuron_2 = Neuron(5, total_spiking_time=tmax, delay=0.04, stop_after=4)
    post_neuron = Neuron(0)
    synapse_1 = Synapse(neuron_1, post_neuron, lr)
    synapse_2 = Synapse(neuron_2, post_neuron, lr, weight=1)

    neuron_1.generate_spike_train(dt)
    neuron_2.generate_spike_train(dt)

    # Simulating the weight change
    list_weight = []

    for t in list_t:
        list_weight.append(synapse_1.weight)
        post_neuron.generate_output_spike_train([synapse_1.weight, synapse_2.weight],
                                                [neuron_1.convolved_spike_train, neuron_2.convolved_spike_train])
        synapse_1.update_weight(learning_rule='iso', delta_t=dt, t=t)

    list_developments.append(list_weight)

for idx, lr in enumerate(list_mu):
    plt.plot(list_t[4001:], list_developments[idx][4001:], label = lr)
plt.title(f"Weight change for Synapse 1 using Iso rule for different learning rates")
plt.xlabel("Time")
plt.ylabel("Weight")
plt.legend()
plt.show()


#Varying dt
list_dt = [0.001, 0.005, 0.01, 0.02, 0.1]
list_developments = []
for dt in list_dt:
    # Defining the system
    neuron_1 = Neuron(5, total_spiking_time=tmax)
    neuron_2 = Neuron(5, total_spiking_time=tmax, delay=0.04, stop_after=4)
    post_neuron = Neuron(0)
    synapse_1 = Synapse(neuron_1, post_neuron, 0.1)
    synapse_2 = Synapse(neuron_2, post_neuron, learning_rate, weight=1)

    neuron_1.generate_spike_train(dt)
    neuron_2.generate_spike_train(dt)

    # Simulating the weight change
    list_weight = []

    for t in list_t:
        list_weight.append(synapse_1.weight)
        post_neuron.generate_output_spike_train([synapse_1.weight, synapse_2.weight],
                                                [neuron_1.convolved_spike_train, neuron_2.convolved_spike_train])
        synapse_1.update_weight(learning_rule='iso', delta_t=dt, t=t)

    list_developments.append(list_weight)

for idx, d in enumerate(list_dt):
    plt.plot(list_t[4001:], list_developments[idx][4001:], label=d)
plt.title(f"Weight change for Synapse 1 using ISO rule for different time steps")
plt.xlabel("Time")
plt.ylabel("Weight")
plt.legend()
plt.show()
