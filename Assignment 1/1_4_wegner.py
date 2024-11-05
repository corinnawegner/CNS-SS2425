import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from learning_rules import Neuron, Synapse
# Exercise 4: Oja Rule with Two Presynaptic Neurons

# Create two presynaptic neurons with different constant currents
pre_neuron_4a = Neuron(constant_current=0.5)
pre_neuron_4b = Neuron(constant_current=0.7)
post_neuron_4 = Neuron(constant_current=0.1)

# Create synapses from the presynaptic neurons to the postsynaptic neuron
synapse4a = Synapse(pre_neuron_4a, post_neuron_4, learning_rate=1, weight=0.1)
synapse4b = Synapse(pre_neuron_4b, post_neuron_4, learning_rate=1, weight=0.1)

# Lists to store data for plotting
list_weight_4a = []
list_weight_4b = []

# Simulate for 100 seconds with 0.1s time step
for t in np.arange(0, 100, 0.1):
    list_weight_4a.append(synapse4a.weight)
    list_weight_4b.append(synapse4b.weight)

    # Update postsynaptic rate (sum of influences)
    post_neuron_4.update_rate(
        [synapse4a.weight, synapse4b.weight],
        [pre_neuron_4a.rate, pre_neuron_4b.rate]
    )

    # Update synapse weights using the Oja rule
    synapse4a.update_weight(learning_rule="oja")
    synapse4b.update_weight(learning_rule="oja")

# Create DataFrame for synapse data
df_data_4a = pd.DataFrame({"time": np.arange(0, 100, 0.1), "weight_4a": list_weight_4a})
df_data_4b = pd.DataFrame({"time": np.arange(0, 100, 0.1), "weight_4b": list_weight_4b})

# Plot the time course of weights for both synapses in Exercise 4
plt.figure(figsize=(12, 8))
plt.plot(df_data_4a['time'], df_data_4a['weight_4a'], label='Constant current: 0.5', color='purple')
plt.plot(df_data_4b['time'], df_data_4b['weight_4b'], label='Constant current: 0.7', color='orange')

# Adding labels and title
plt.xlabel('Time')
plt.ylabel('Weight')
plt.title('Time Course of Synaptic Weights using Oja Rule for Two Presynaptic Neurons')
plt.grid(True)
plt.legend()
plt.show()
