# Import matplotlib library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_excel('data7/sac_0%_0.7435.xlsx', header=None)

# Generate some sample data for x and y
x = [i for i in range(100)]
y = [0.1 * i + 0.5 * (i ** 0.5) for i in x]

# Calculate the smoothed curve using a moving average
window_size = 10
smoothed_y = []
for i in range(len(y)):
    start = max(0, i - window_size // 2)
    end = min(len(y), i + window_size // 2)
    smoothed_y.append(sum(y[start:end]) / (end - start))

# Calculate the upper and lower bounds of the variance using standard deviation
import statistics
std_y = statistics.stdev(y)
upper_y = [y + std_y for y in smoothed_y]
lower_y = [y - std_y for y in smoothed_y]

# Plot the data and the curves
plt.plot(x, y, 'o', color='grey', label='Data')
plt.plot(x, smoothed_y, '-', color='blue', label='Smoothed curve')
plt.fill_between(x, lower_y, upper_y, color='lightblue', label='Variance')
plt.xlabel('Training episodes')
plt.ylabel('Reward')
plt.legend()
plt.show()