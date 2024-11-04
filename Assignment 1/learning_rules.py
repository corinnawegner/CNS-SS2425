import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, constant_current):
        self.constant_current = constant_current
        self.rate = constant_current

    def update_rate(self, presynaptic_weights, input_rates):
        self.rate = np.sum(np.array(presynaptic_weights)*np.array(input_rates)) + self.constant_current

class Synapse:
    def __init__(self, in_neuron, out_neuron, learning_rate, weight=0):
        self.weight = weight
        self.learning_rate = learning_rate
        self.in_neuron = in_neuron
        self.out_neuron = out_neuron

    def update_weight(self, learning_rule="hebb", delta_t=0.1):
        rate_in = self.in_neuron.rate
        rate_out = self.out_neuron.rate
        if learning_rule == "hebb":
            self.weight += self.hebb_rule(rate_in, rate_out, delta_t)
        elif learning_rule == "bcm":
            self.weight += self.bcm_rule(rate_in, rate_out, delta_t)
        elif learning_rule == "oja":
            self.weight += self.oja_rule(rate_in, rate_out, delta_t)
        else:
            raise ValueError(f"Unknown learning rule: {learning_rule}")
    def hebb_rule(self, rate_in, rate_out, delta_t):
        delta_weight = self.learning_rate * rate_in * rate_out * delta_t
        return delta_weight

    def bcm_rule(self, rate_in, rate_out, delta_t):
        delta_weight = self.learning_rate * rate_in * rate_out * delta_t * (rate_out - 0.3)
        return delta_weight

    def oja_rule(self, rate_in, rate_out, delta_t):
        delta_weight = self.learning_rate*(rate_in * rate_out - self.weight * rate_in**2)* delta_t
        return delta_weight
# Exercise 2

pre_neuron = Neuron(constant_current=0.5)
post_neuron = Neuron(constant_current=0.05)
synapse1 = Synapse(pre_neuron, post_neuron,0.1)

list_t = []
list_rate_pre = []
list_rate_post = []
list_weight = []

for t in np.arange(0, 100, 0.1):
    time = t
    list_t.append(time)
    list_rate_pre.append(pre_neuron.rate)
    list_rate_post.append(post_neuron.rate)
    list_weight.append(synapse1.weight)
    post_neuron.update_rate([synapse1.weight], [pre_neuron.rate])
    synapse1.update_weight()

df_data = pd.DataFrame({"time": list_t, "rate_j": list_rate_pre, "rate_in": list_rate_post, "weight": list_weight})

print(df_data.head())

plt.figure(figsize=(10, 6))
plt.plot(df_data['time'], df_data['weight'], label='Synaptic Weight', color='blue')
plt.xlabel('Time')
plt.ylabel('Weight')
plt.title('Time Course of Synaptic Weight')
plt.grid(True)
plt.legend()
plt.show()

# Exercise 3

pre_neuron_2 = Neuron(constant_current=0.5)
post_neuron_2 = Neuron(constant_current=0.2)
synapse2 = Synapse(pre_neuron, post_neuron,0.1)

pre_neuron_3 = Neuron(constant_current=0.5)
post_neuron_3 = Neuron(constant_current=0.4)
synapse3 = Synapse(pre_neuron, post_neuron,0.1)

list_rate_pre_2 = []
list_rate_pre_3 = []
list_rate_post_2 = []
list_rate_post_3 = []
list_weight_2 = []
list_weight_3 = []

for t in np.arange(0, 100, 0.1):
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
plt.plot(df_data_2['time'], df_data_2['weight_2'], label='Synaptic Weight 2', color='red')
plt.plot(df_data_3['time'], df_data_3['weight_3'], label='Synaptic Weight 3', color='green')

# Adding labels and title
plt.xlabel('Time')
plt.ylabel('Weight')
plt.title('Time Course of Synaptic Weights for Synapse 2 and Synapse 3')
plt.grid(True)
plt.legend()
plt.show()

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
plt.plot(df_data_4a['time'], df_data_4a['weight_4a'], label='Synaptic Weight 4a (Current 0.5)', color='purple')
plt.plot(df_data_4b['time'], df_data_4b['weight_4b'], label='Synaptic Weight 4b (Current 0.7)', color='orange')

# Adding labels and title
plt.xlabel('Time')
plt.ylabel('Weight')
plt.title('Time Course of Synaptic Weights using Oja Rule for Two Presynaptic Neurons')
plt.grid(True)
plt.legend()
plt.show()




