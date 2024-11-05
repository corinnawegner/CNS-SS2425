import numpy as np
from sklearn.decomposition import PCA

path_data = r"C:\Users\corin\PycharmProjects\CNS-SS2425\cmake-build-debug\random_numbers.txt"

data = np.loadtxt(path_data, delimiter='\t')

mean = np.mean(data, axis=0)
print(mean)

pca = PCA(n_components=2)
pca.fit(data)

components = pca.components_
explained_variance = pca.explained_variance_

print(components[0])