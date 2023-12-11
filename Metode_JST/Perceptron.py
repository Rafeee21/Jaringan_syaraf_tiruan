#Mengunakan metode perceptron
import numpy as np

# Fungsi aktivasi
def step_function(x):
    return np.where(x<=0, 0, 1)

# Data input dan target
buah = np.array([[8.6,2.7], [7,3.2], [7.7,2.9], [8.4,3], [6.7,2.5], [8.1,7.8], [6.5,6.8], [6.5,6.5], [6.2,6.4], [6.6,7.1]])
target = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])

# Inisialisasi bobot dan bias
weights = np.random.uniform(size=(2,2))

# Laju pembelajaran
lr = 0.9

# Pelatihan JST
for epoch in range(10):
    for i in range(len(buah)):
        # Feedforward
        input_layer = buah[i]
        output_layer = step_function(np.dot(input_layer, weights))

        # Perhitungan kesalahan
        error = target[i] - output_layer

        # Perubahan bobot
        weights += lr * np.outer(error, input_layer)

        # Menampilkan data, iterasi, dan perubahan bobot
        print(f'Epoch: {epoch+1}')
        print(f'Error: {np.mean(np.abs(error))}')
        print(f'Weights: {weights}')
        print(f'Output: {output_layer}')
