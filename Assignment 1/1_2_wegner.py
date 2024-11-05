from learning_rules import Neuron, Synapse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
