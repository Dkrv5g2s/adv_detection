import numpy as np
import matplotlib.pyplot as plt

class TriangularFuzzySets:
    def __init__(self, centers=None, width=None):
        if centers is None:
            centers = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        if width is None:
            width = np.array([0.3, 0.25, 0.25, 0.25, 0.3])

        self.centers = centers
        self.widths = width
        self.K = len(centers)
        self.labels = ["Very Low", "Low", "OK", "High", "Very High"]

    def membership(self, x):
        x_expand = np.expand_dims(x, axis=-1)
        c = self.centers.reshape((1,) * (x_expand.ndim - 1) + (self.K,))
        w = self.widths.reshape((1,) * (x_expand.ndim - 1) + (self.K,))
        mu = np.maximum(0, 1 - np.abs(x_expand - c) / (w + 1e-12))
        return mu

# Create and plot
fuzzy_sets = TriangularFuzzySets()
x = np.linspace(0, 1, 1000)
memberships = fuzzy_sets.membership(x)

plt.figure(figsize=(10, 6))
colors = ['red', 'orange', 'green', 'blue', 'purple']

for i in range(fuzzy_sets.K):
    plt.plot(x, memberships[:, i], color=colors[i], linewidth=2,
             label=fuzzy_sets.labels[i])

plt.xlabel('Input Value')
plt.ylabel('Membership Degree')
plt.title('Triangular Fuzzy Sets')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('fuzzy_sets.png', dpi=300, bbox_inches='tight')
plt.show()
