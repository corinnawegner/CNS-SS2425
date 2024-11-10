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

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

for key, rn in dict_r.items():
    # Initialize neurons and synapses
    pre_neuron_6a = Neuron(rn[0][0])
    pre_neuron_6b = Neuron(rn[0][1])
    post_neuron_6 = Neuron(0.1)

    synapse_6a = Synapse(pre_neuron_6a, post_neuron_6, 0.005, 0)
    synapse_6b = Synapse(pre_neuron_6b, post_neuron_6, 0.005, 0)

    list_weight_6a = []
    list_weight_6b = []

    # Simulate and record weights over time
    for t in np.arange(0, 2000, 0.1):
        list_weight_6a.append(synapse_6a.weight)
        list_weight_6b.append(synapse_6b.weight)

        post_neuron_6.update_rate(
            [synapse_6a.weight, synapse_6b.weight],
            [pre_neuron_6a.rate, pre_neuron_6b.rate]
        )

        synapse_6a.update_weight()
        synapse_6b.update_weight()

        if t % 2 == 0:
            pre_neuron_6a.constant_current = rn[int(t / 2), 0]
            pre_neuron_6b.constant_current = rn[int(t / 2), 1]

    # Create DataFrames for weights
    df_data_6a = pd.DataFrame({"time": np.arange(0, 2000, 0.1), "weight_6a": list_weight_6a})
    df_data_6b = pd.DataFrame({"time": np.arange(0, 2000, 0.1), "weight_6b": list_weight_6b})

    # Calculate PCA on random numbers
    mean_num = np.mean(rn, axis=0)
    pca = PCA(n_components=1)
    pca.fit(rn)
    dir_var = pca.components_[0]

    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # First subplot: Scatter plot for weights
    ax1.scatter(df_data_6a['weight_6a'], df_data_6b['weight_6b'], color='blue', label='Weights')
    ax1.set_xlabel('Weight a')
    ax1.set_ylabel('Weight b')
    ax1.set_title(f'Weights plotted against each other for {key}')
    ax1.grid(True)
    ax1.legend()

    # Second subplot: Scatter plot for random numbers with PCA component
    ax2.scatter(rn[:, 0], rn[:, 1], label="Random Numbers", color="grey", alpha=0.5)
    ax2.quiver(mean_num[0], mean_num[1], dir_var[0], dir_var[1],
               angles='xy', scale_units='xy', scale=1, color='green', label="Smallest Variance Direction")
    ax2.set_xlabel('Random Numbers 1')
    ax2.set_ylabel('Random Numbers 2')
    ax2.set_title(f'Random Numbers with PCA for {key}')
    ax2.grid(True)
    ax2.legend()

    # Display the plots
    plt.suptitle(f'Weights and Random Numbers Analysis for {key}', fontsize=16)
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
for t in np.arange(0, 2000, 0.1):
    if t % 2 == 0:
        pre_neuron_6c.constant_current = random_numbers_4[int(t / 2), 0]
        pre_neuron_6d.constant_current = random_numbers_4[int(t / 2), 1]
        pre_neuron_6c.rate = random_numbers_4[int(t / 2), 0]
        pre_neuron_6d.rate = random_numbers_4[int(t / 2), 1]

#    print(pre_neuron_6c.rate, pre_neuron_6d.rate)

    list_weight_6c.append(synapse6c.weight)
    list_weight_6d.append(synapse6d.weight)

    # Update postsynaptic rate (sum of influences)
    post_neuron_62.update_rate(
        [synapse6c.weight, synapse6d.weight],
        [pre_neuron_6c.rate, pre_neuron_6d.rate]
    )

    # Update synapse weights using the covariance rule
    synapse6c.update_weight(learning_rule="cov")
    synapse6d.update_weight(learning_rule="cov")

df_data_6c = pd.DataFrame({"weight_6c": list_weight_6c})
df_data_6d = pd.DataFrame({"weight_6d": list_weight_6d})

# Calculate the mean and principal component for the random numbers
mean_num = np.mean(random_numbers_4, axis=0)
pca = PCA(n_components=1)
pca.fit(random_numbers_4)
dir_var = pca.components_[0]

# Create main figure and two subplots arranged horizontally
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# First subplot: Scatter plot for weights
ax1.scatter(df_data_6c['weight_6c'], df_data_6d['weight_6d'], color='blue', label='Weights')
ax1.set_xlabel('Weight 1', color='blue')
ax1.set_ylabel('Weight 2', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Adjust y-axis limits for the first plot based on the weights
weight_6d_min, weight_6d_max = df_data_6d['weight_6d'].min(), df_data_6d['weight_6d'].max()
ax1.set_ylim(weight_6d_min, weight_6d_max)

# Title, grid, and legend for the first subplot
ax1.set_title('Weights Comparison')
ax1.grid(True)
ax1.legend()

# Second subplot: Scatter plot for random numbers and PCA component
ax2.scatter(random_numbers_4[:, 0], random_numbers_4[:, 1], color="grey", alpha=0.5, label="Random Numbers")
ax2.set_xlabel('Random Numbers 1', color='grey')
ax2.set_ylabel('Random Numbers 2', color='grey')
ax2.tick_params(axis='x', labelcolor='grey')
ax2.tick_params(axis='y', labelcolor='grey')

# Adjust limits for the second plot based on the random numbers
ax2.set_xlim(random_numbers_4[:, 0].min() - 1, random_numbers_4[:, 0].max() + 1)
ax2.set_ylim(random_numbers_4[:, 1].min() - 1, random_numbers_4[:, 1].max() + 1)

# Add PCA direction as a quiver plot on the second subplot
ax2.quiver(mean_num[0], mean_num[1], dir_var[0], dir_var[1],
           angles='xy', scale_units='xy', scale=1, color='green', label="Principal component")

# Title, grid, and legend for the second subplot
ax2.set_title('Random Numbers with Principal Component')
ax2.grid(True)
ax2.legend()

# Display the plot
plt.suptitle('Weights and Constant currents for the Covariance Rule', fontsize=16)
plt.show()