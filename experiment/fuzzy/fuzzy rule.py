import numpy as np
import matplotlib.pyplot as plt

# Define membership functions for input fuzzy sets A1 and A2
def mu_A1(x):
    if x <= 18:
        return 1
    elif 18 < x <= 20:
        return (20 - x) / 2
    else:
        return 0

def mu_A2(x):
    if 20 <= x <= 25:
        return (x - 20) / 5
    else:
        return 0

# Define membership functions for output fuzzy sets B1 and B2
def mu_B1(y):
    if y <= 1:
        return 1
    elif 1 < y <= 2:
        return 2 - y
    else:
        return 0

def mu_B2(y):
    if y <= 2:
        return 0
    elif 2 < y <= 5:
        return (y - 2) / 3
    else:
        return 1

# Truncate the membership function of B1 based on alpha
def truncated_B1(y, alpha):
    return min(mu_B1(y), alpha)

# Initialize input x and calculate membership degrees
x_input = 19
mu_A1_x = mu_A1(x_input)
mu_A2_x = mu_A2(x_input)

# Define the range for y
y_range = np.linspace(0, 5, 500)

# Compute truncated B1 values
truncated_B1_values = [truncated_B1(y, mu_A1_x) for y in y_range]

# Perform defuzzification using the centroid method
defuzz_numerator = np.sum(y_range * truncated_B1_values)
defuzz_denominator = np.sum(truncated_B1_values)
centroid = defuzz_numerator / defuzz_denominator

# Plotting
plt.figure(figsize=(12, 8))

# Input fuzzy sets A1 and A2
x_range = np.linspace(15, 30, 500)
plt.subplot(2, 2, 1)
plt.plot(x_range, [mu_A1(x) for x in x_range], label="$A_1$: Cold")
plt.plot(x_range, [mu_A2(x) for x in x_range], label="$A_2$: Comfortable")
plt.scatter([x_input], [mu_A1_x], color="red", label=f"Input x = {x_input}, $A_1$ membership degree = {mu_A1_x:.2f}")
plt.scatter([x_input], [mu_A2_x], color="blue", label=f"Input x = {x_input}, $A_2$ membership degree = {mu_A2_x:.2f}")
plt.title("Input Fuzzy Sets A")
plt.xlabel("Room Temperature x (°C)")
plt.ylabel("Membership Degree")
plt.legend()
plt.grid()

# Output fuzzy set B1 and truncated B1
plt.subplot(2, 2, 2)
plt.plot(y_range, [mu_B1(y) for y in y_range], label="$B_1$: Low Wind Speed")
plt.plot(y_range, truncated_B1_values, label="Truncated $B_1$", linestyle="--")
plt.title("Output Fuzzy Set B1")
plt.xlabel("Air Conditioner Wind Speed y")
plt.ylabel("Membership Degree")
plt.legend()
plt.grid()

# Output fuzzy set B2
plt.subplot(2, 2, 3)
plt.plot(y_range, [mu_B2(y) for y in y_range], label="$B_2$: High Wind Speed")
plt.title("Output Fuzzy Set B2")
plt.xlabel("Air Conditioner Wind Speed y")
plt.ylabel("Membership Degree")
plt.legend()
plt.grid()

# Defuzzification result
plt.subplot(2, 2, 4)
plt.plot(y_range, truncated_B1_values, label="Truncated $B_1$", linestyle="--")
plt.axvline(centroid, color="red", linestyle="--", label=f"Defuzzification Result y = {centroid:.2f}")
plt.title("Defuzzification")
plt.xlabel("Air Conditioner Wind Speed y")
plt.ylabel("Membership Degree")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("fuzzy_control_visualization.png")
plt.show()
