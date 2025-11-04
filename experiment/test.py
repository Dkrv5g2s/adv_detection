import numpy as np
import matplotlib.pyplot as plt

# Define f1(x, y) and f2(x, y)
f1 = np.array([[2, 1, 3],
               [4, 3, 2],
               [2, 2, 1]])
f2 = np.array([[2, 3],
               [1, 1]])

# Convolution: flip f2 and compute
f2_flipped = np.flip(np.flip(f2, axis=0), axis=1)
convolution_result = np.zeros((f1.shape[0] + f2.shape[0] - 1, f1.shape[1] + f2.shape[1] - 1))

for i in range(convolution_result.shape[0]):
    for j in range(convolution_result.shape[1]):
        for m in range(f2_flipped.shape[0]):
            for n in range(f2_flipped.shape[1]):
                if 0 <= i - m < f1.shape[0] and 0 <= j - n < f1.shape[1]:
                    convolution_result[i, j] += f1[i - m, j - n] * f2_flipped[m, n]

# Correlation: directly compute without flipping
correlation_result = np.zeros_like(convolution_result)

for i in range(correlation_result.shape[0]):
    for j in range(correlation_result.shape[1]):
        for m in range(f2.shape[0]):
            for n in range(f2.shape[1]):
                if 0 <= i - m < f1.shape[0] and 0 <= j - n < f1.shape[1]:
                    correlation_result[i, j] += f1[i - m, j - n] * f2[m, n]

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Convolution Result")
plt.imshow(convolution_result, cmap='gray', interpolation='none')
plt.colorbar()
plt.gca().invert_yaxis()

plt.subplot(1, 2, 2)
plt.title("Correlation Result")
plt.imshow(correlation_result, cmap='gray', interpolation='none')
plt.colorbar()
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()