# Exercise 6
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from learning_rules import Neuron, Synapse
from sklearn.decomposition import PCA

path_data = r"C:\Users\corin\PycharmProjects\CNS-SS2425\cmake-build-debug\random_numbers.txt"
random_numbers = np.loadtxt(path_data, delimiter='\t')
random_numbers_2 = np.loadtxt(r"C:\Users\corin\PycharmProjects\CNS-SS2425\cmake-build-debug\random_numbers_20_0.txt", delimiter='\t')
random_numbers_3 = np.loadtxt(r"C:\Users\corin\PycharmProjects\CNS-SS2425\cmake-build-debug\random_numbers_45_0.txt", delimiter='\t')
random_numbers_4 = np.loadtxt(r"C:\Users\corin\PycharmProjects\CNS-SS2425\cmake-build-debug\random_numbers_45_2.txt", delimiter='\t')

list_r = [random_numbers, random_numbers_2, random_numbers_3, random_numbers_4]
vals_r = ["Phi: 0, Offset: (0,0)", "Phi: 20°, Offset: (0,0)", "Phi: 45, Offset: (0,0)", "Phi: -45°, Offset: (2,2)"]
dict_r = {vals_r[i]: list_r[i] for i in range(len(vals_r))}

for key, rn in dict_r.items():
    pre_neuron_6a = Neuron(rn[0][0])
    pre_neuron_6b = Neuron(rn[0][1])
    post_neuron_6 = Neuron(0.1)

    synapse_6a = Synapse(pre_neuron_6a, post_neuron_6, 0.005, 0)
    synapse_6b = Synapse(pre_neuron_6b, post_neuron_6, 0.005, 0)

    list_weight_6a = []
    list_weight_6b = []

    for t in np.arange(0, 2000, 0.1):
        list_weight_6a.append(synapse_6a.weight)
        list_weight_6b.append(synapse_6b.weight)

        post_neuron_6.update_rate(
            [synapse_6a.weight, synapse_6b.weight],
            [pre_neuron_6a.rate, pre_neuron_6b.rate]
        )

        synapse_6a.update_weight()
        synapse_6b.update_weight()

        if t%2 == 0:
            pre_neuron_6a.constant_current = random_numbers[int(t/2), 0]
            pre_neuron_6b.constant_current = random_numbers[int(t/2), 1]


    df_data_6a = pd.DataFrame({"time": np.arange(0, 2000, 0.1), "weight_6a": list_weight_6a})
    df_data_6b = pd.DataFrame({"time": np.arange(0, 2000, 0.1), "weight_6b": list_weight_6b})

    plt.figure(figsize=(12, 8))
    #plt.plot(np.arange(0, 2000, 2), random_numbers[:,1], color="darkgrey", label= "Constant currents")
    #plt.plot(np.arange(0, 2000, 2), random_numbers[:,0], color="lightgrey", label= "Constant currents")
    #plt.plot([0,2000],[list_weight_6a[0], list_weight_6a[-1]], color = "black")
    #plt.plot([0,2000],[list_weight_6b[0], list_weight_6b[-1]], color = "black")
    #plt.plot(df_data_6a['time'], df_data_6a['weight_6a'], label='Synaptic Weight for Synapse 1', color='purple')
    #plt.plot(df_data_6b['time'], df_data_6b['weight_6b'], label='Synaptic Weight for Synapse 2', color='orange')
    plt.scatter(df_data_6a['weight_6a'], df_data_6b['weight_6b'], color='blue',label='Weights')
    plt.scatter(random_numbers[:, 0], random_numbers[:, 1], label="random numbers", color="grey", alpha=0.5)

    mean_num = np.mean(random_numbers, axis=0)
    pca = PCA(n_components=1)
    pca.fit(random_numbers)
    dir_var = pca.components_[0]
    plt.quiver(mean_num[0], mean_num[1], dir_var[0], dir_var[1],
               angles='xy', scale_units='xy', scale=1, color='green', label="Smallest Variance Direction")

    # Adding labels and title
    plt.xlabel('Weight a')
    plt.ylabel('Weight b')
    plt.title(f'Weights plotted against each other for {key}')
    plt.grid(True)
    plt.legend()
    plt.show()

# Covariance rule

pre_neuron_6c = Neuron(constant_current=random_numbers_4[0,0])
pre_neuron_6d = Neuron(constant_current=random_numbers_4[0,1])
post_neuron_62 = Neuron(constant_current=0.1)

# Create synapses from the presynaptic neurons to the postsynaptic neuron
synapse6c = Synapse(pre_neuron_6c, post_neuron_62, learning_rate=0.005, weight=0)
synapse6d = Synapse(pre_neuron_6d, post_neuron_62, learning_rate=0.005, weight=0)

# Lists to store data for plotting
list_weight_6c = []
list_weight_6d = []

# Simulate for 100 seconds with 0.1s time step
for t in np.arange(0, 100, 0.1):
    if t % 2 == 0:
        pre_neuron_6c.constant_current = random_numbers[int(t / 2), 0]
        pre_neuron_6d.constant_current = random_numbers[int(t / 2), 1]

    pre_neuron_6c.all_rates.append(pre_neuron_6c.constant_current)
    pre_neuron_6d.all_rates.append(pre_neuron_6d.constant_current)

    list_weight_6c.append(synapse6c.weight)
    list_weight_6d.append(synapse6d.weight)

    # Update postsynaptic rate (sum of influences)
    post_neuron_62.update_rate(
        [synapse6c.weight, synapse6d.weight],
        [pre_neuron_6c.rate, pre_neuron_6d.rate]
    )

    # Update synapse weights using the Oja rule
    synapse6c.update_weight(learning_rule="cov")
    synapse6d.update_weight(learning_rule="cov")

# Create DataFrame for synapse data
df_data_6c = pd.DataFrame({"time": np.arange(0, 100, 0.1), "weight_6c": list_weight_6c})
df_data_6d = pd.DataFrame({"time": np.arange(0, 100, 0.1), "weight_6d": list_weight_6d})

# Plot the time course of weights for both synapses in Exercise 4
plt.figure(figsize=(12, 8))
#plt.plot(df_data_6c['time'], df_data_6c['weight_6c'], label='Synaptic Weight 6c', color='purple')
#plt.plot(df_data_6d['time'], df_data_6d['weight_6d'], label='Synaptic Weight 6d', color='orange')

plt.scatter(df_data_6c['weight_6c'], df_data_6d['weight_6d'], color='blue', label='Weights')
plt.scatter(random_numbers_4[:, 0], random_numbers_4[:, 1], label="random numbers", color="grey", alpha=0.5)

# Adding labels and title
plt.xlabel('Time')
plt.ylabel('Weight')
plt.title('Weights plotted against each other using Covariance Rule for Two Presynaptic Neurons')
plt.grid(True)
plt.legend()
plt.show()
