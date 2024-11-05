# Exercise 3
from learning_rules import Neuron, Synapse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pre_neuron_2 = Neuron(constant_current=0.5)
post_neuron_2 = Neuron(constant_current=0.2)
synapse2 = Synapse(pre_neuron_2, post_neuron_2,0.1)

pre_neuron_3 = Neuron(constant_current=0.5)
post_neuron_3 = Neuron(constant_current=0.4)
synapse3 = Synapse(pre_neuron_3, post_neuron_3,0.1)

list_rate_pre_2 = []
list_rate_pre_3 = []
list_rate_post_2 = []
list_rate_post_3 = []
list_weight_2 = []
list_weight_3 = []

list_t = np.arange(0, 100, 0.1)

for t in list_t:
    list_rate_pre_2.append(pre_neuron_2.rate)
    list_rate_pre_3.append(pre_neuron_3.rate)
    list_rate_post_2.append(post_neuron_2.rate)
    list_rate_post_3.append(post_neuron_3.rate)
    list_weight_2.append(synapse2.weight)
    list_weight_3.append(synapse3.weight)
    post_neuron_2.update_rate([synapse2.weight], [pre_neuron_2.rate])
    post_neuron_3.update_rate([synapse3.weight], [pre_neuron_3.rate])
    synapse2.update_weight(learning_rule="bcm")
    synapse3.update_weight(learning_rule="bcm")

# Create DataFrame for synapse data
df_data_2 = pd.DataFrame({"time": list_t, "weight_2": list_weight_2})
df_data_3 = pd.DataFrame({"time": list_t, "weight_3": list_weight_3})

# Plot the time course of weights for all synapses
plt.figure(figsize=(12, 8))
plt.plot(df_data_2['time'], df_data_2['weight_2'], label='constant current: 0.2', color='red')
plt.plot(df_data_3['time'], df_data_3['weight_3'], label='constant current: 0.3', color='green')

# Adding labels and title
plt.xlabel('Time')
plt.ylabel('Weight')
plt.title('Time course of synaptic weights with BCM rule')
plt.grid(True)
plt.legend()
plt.show()