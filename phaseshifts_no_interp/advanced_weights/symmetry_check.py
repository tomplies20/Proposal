import matplotlib.pyplot as plt
import numpy as np

num_points = 100

x = np.linspace(0, 100, num_points)

matrix = np.ones((num_points, num_points))




'''
for i in range(num_points):
    for j in range(i+1, num_points):
        matrix[i, j] = matrix[i, j] * i* 0.5
        matrix[j, i] = matrix[i, j]
'''
for i in range(num_points):
    for j in range( num_points):
        matrix[i, j] = matrix[i, j] * i* 0.5

plt.matshow(matrix)
plt.show()