import numpy as np
import matplotlib.pyplot as plt



'''
plt.plot(x, np.sqrt(np.sqrt(x)))
plt.plot(x, x)
plt.show()
'''
# grid_size
x_weight = np.linspace(1, 0, 200)

num_points = 15000

#dim = num_points x grid_size
phaseshifts = np.array()

#dim = num_points
weights = np.zeros((num_points))



in_out_array = np.zeros((num_points, grid_size))
for m in range(num_points):
    for n in range(grid_size):
        if reference_interpolated[n] - np.abs(error_interpolated[n]) <= phaseshift_set[m, n] <= reference_interpolated[
            n] + np.abs(error_interpolated[n]):
            in_out_array[m, n] = 1

weights = np.dot(in_out_array, x_weight)



