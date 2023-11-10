import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import random

import pymc as pm
from pymc.gp.cov import *
from pymc.gp.util import stabilize

# import tensorflow as tf


energy_ = np.array([7.46471504e-02, 6.71824354e-01, 1.86617876e+00, 3.65771037e+00,
                    6.04641918e+00, 9.03230520e+00, 1.26153684e+01, 1.67956088e+01,
                    2.15730265e+01, 2.69476213e+01, 3.29193933e+01, 3.94883426e+01,
                    4.66544690e+01, 5.44177727e+01, 6.27782535e+01, 7.17359115e+01,
                    8.12907468e+01, 9.14427593e+01, 1.02191949e+02, 1.13538316e+02,
                    1.25481860e+02, 1.38022581e+02, 1.51160480e+02, 1.64895555e+02,
                    1.79227808e+02, 1.94157238e+02, 2.09683846e+02, 2.25807630e+02,
                    2.42528592e+02, 2.59846731e+02, 2.77762047e+02, 2.96274540e+02,
                    3.15384211e+02, 3.35091058e+02, 3.55395083e+02, 3.76296285e+02,
                    3.97794665e+02, 4.19890221e+02, 4.42582955e+02, 4.65872866e+02,
                    4.89759954e+02, 5.14244219e+02, 5.39325662e+02, 5.65004281e+02,
                    5.91280078e+02, 6.18153053e+02, 6.45623204e+02, 6.73690532e+02,
                    7.02355038e+02, 7.31616721e+02, 7.61475581e+02, 7.91931619e+02,
                    8.22984833e+02, 8.54635225e+02, 8.86882794e+02, 9.19727540e+02,
                    9.53169464e+02, 9.87208564e+02, 1.02184484e+03, 1.05707830e+03,
                    1.09290893e+03, 1.12933674e+03, 1.16636173e+03, 1.20398389e+03,
                    1.24220323e+03, 1.28101975e+03, 1.32043344e+03, 1.36044432e+03,
                    1.40105237e+03, 1.44225759e+03, 1.48406000e+03, 1.52645958e+03,
                    1.56945634e+03, 1.61305027e+03, 1.65724139e+03, 1.70202968e+03,
                    1.74741514e+03, 1.79339779e+03, 1.83997761e+03, 1.88715461e+03,
                    1.93492879e+03, 1.98330014e+03, 2.03226867e+03, 2.08183438e+03,
                    2.13200416e+03, 2.18377803e+03, 2.23715698e+03, 2.29214100e+03,
                    2.34873010e+03, 2.40692428e+03, 2.46672353e+03, 2.52812786e+03,
                    2.59113726e+03, 2.65575173e+03, 2.72197128e+03, 2.78979590e+03,
                    2.85922559e+03, 2.93026036e+03, 3.00290020e+03, 3.07714512e+03])

grid_size = 200
upper_energy_limit = 200
energy = np.linspace((energy_[0] / upper_energy_limit) ** (1 / 3), 1, grid_size) ** 3 * upper_energy_limit

triton_data = np.loadtxt('triton_results_x.txt')


def gaussian(x, mu, sigma):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-1 / 2 * ((x - mu) / sigma) ** 2)

data_size = 480
new_data_size = 480
# parameters
lin_size_max = 200
all_weights = np.zeros((data_size, lin_size_max))

lin_size = 5
l = 0.000001
l=5
new_upper_limit = 200

energy_lin_200 = np.linspace(energy[0], 200, 200)

#energy_lin = np.linspace(0.1, 6, 20)

#energy_lin = np.array([ 10, 20, 50, 100, 175])
#energy_lin = np.array([2, 10, 20, 50, 100])


reference_data = np.loadtxt(
    f'/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/phaseshift_files/phaseshifts_SVD/phaseshifts_unchanged_SLLJT_{partial_wave}_lambda_2.00_s5.dat')
reference = reference_data[:, 3]

# energy_lin = np.array([energy_lin_200[1], energy_lin_200[10], energy_lin_200[20], energy_lin_200[50], energy_lin_200[100]])



partial_wave = '10010'

data_ = np.loadtxt(
    '/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/NNs/LEC_ratio/LEC_files/phaseshifts_SLLJT_10010_lambda_2.00_s5_new_x.dat')

LECs = data_[:, 0:5]

phaseshifts = data_[:, 5:205]

errors = np.loadtxt(
    f'/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/phaseshift_files/EKM_uncertainty/phaseshifts_uncertainties_SLLJT_{partial_wave}_lambda_2.00_s5.dat')
N3LO_error = errors[:, 3]


def kernel(energy_lin, i, j):
    distance = abs(energy_lin[i] - energy_lin[j])
    print('energy i: ' + str(energy_lin[i]))
    print('energy j: ' + str(energy_lin[j]))
    print('distance ' + str(distance))
    return np.exp(-distance ** 2 / (2 * l ** 2))

def weights_at_energy(en):
    #energy_lin = np.array([en]) #use this for scalar single point energy values
    energy_lin = en

    lin_size = len(energy_lin)

    phaseshifts_interp = np.zeros((len(phaseshifts[:, 0]), lin_size))
    for k in range(len(phaseshifts[:, 0])):
        ps_interpolate = sc.interpolate.interp1d(energy_lin_200, phaseshifts[k, :])
        phaseshifts_interp[k] = ps_interpolate(energy_lin)



    error_interpolate = sc.interpolate.interp1d(energy_, N3LO_error)
    error_interpolated = error_interpolate(energy_lin)


    reference_interpolate = sc.interpolate.interp1d(energy_, reference)
    reference_interpolated = reference_interpolate(energy_lin)
    covariance_matrix = np.zeros((lin_size, lin_size))
    signal_variance = np.abs(error_interpolated)
    mean = reference_interpolated


    # add nugget
    nugget = 1e-10

    for i in range(lin_size):
        for j in range( lin_size):
            covariance_matrix[i, j] =  (signal_variance[i]**2 + signal_variance[j]**2)/2 * kernel(energy_lin, i, j)
            if i == j:
                covariance_matrix[i, j] += nugget


    #for i in range(lin_size - 1):
     #   print(covariance_matrix[i, i + 1])

    mvn = multivariate_normal(mean=mean, cov=covariance_matrix, allow_singular=True)
    # Calculate the probability density at the point of interest
    likelihood = np.zeros((len(phaseshifts_interp[:, 0])))
    for b in range(len(phaseshifts_interp[:, 0])):
        point_of_interest = phaseshifts_interp[b, :]
        pdf = mvn.pdf(point_of_interest)

        likelihood[b] = pdf

    weights = likelihood / np.sum(likelihood)
    all_weights = weights

    return all_weights



def triton(weights):



    weights_ = weights**1

    data_weight_pairs = list(zip(triton_data, weights_))

    # Sort the data based on values while preserving associated weights
    sorted_data_weight_pairs = sorted(data_weight_pairs, key=lambda x: x[0])

    # Calculate the total weight
    total_weight = sum(weight for _, weight in sorted_data_weight_pairs)

    # Calculate indices for weighted percentiles
    target_percentiles = [25, 50, 75]
    percentiles = {}

    for percentile in target_percentiles:
        target_weight = (percentile / 100) * total_weight
        current_weight = 0
        for value, weight in sorted_data_weight_pairs:
            current_weight += weight
            if current_weight >= target_weight:
                percentiles[percentile] = value
                break
    print(percentiles)
    # print("Weighted 25th Percentile:", percentiles[25])
    # print("Weighted 75th Percentile:", percentiles[75])

    percentile_25_bin_edge = percentiles[25]
    percentile_50_bin_edge = percentiles[50]
    percentile_75_bin_edge = percentiles[75]

    median = percentile_50_bin_edge
    width = percentile_75_bin_edge - percentile_25_bin_edge
    mean_= np.average(triton_data, axis=0, weights=weights/max(weights) )
    cov_ = np.cov(triton_data, rowvar=1, aweights=weights / max(weights))
    return median, mean_

ens = np.linspace(1, 50, 99)
ens = np.linspace(0.1, 1.5, 100)
means = np.zeros(len(ens))
medians= np.zeros(len(ens))
covs = np.zeros(len(ens))
for i, en in enumerate(ens):
    weights = weights_at_energy(en)
    medians[i], means[i] = triton(weights)

plt.plot(ens, means, label='means')
#plt.xlabel('E (Mev)')
#plt.legend()
#plt.show()
plt.plot(ens, medians, label='medians')
plt.xlabel('E (Mev)')
plt.legend()
plt.show()

initial_parameters = np.array([])
def objective(energy_lin, optimal_vals):
    weights = weights_at_energy(energy_lin)
    current_vals = triton(weights)
    opt_mean, opt_cov = optimal_vals
    curr_mean, curr_cov = current_vals
    return np.sum(opt_mean - curr_mean)**2 + np.sum(opt_cov - curr_cov)**2

param_bounds = [(0, 5), (5, 10), (10, 25), (25, 50), (50, 100)]

result = minimize(objective, initial_parameters, bounds=param_bounds, constraints=constraints)

# Extract the optimized parameters
optimized_parameters = result.x

'''
diff = np.inf
while(diff > 0.1):
    range_1 = random.uniform(0, 5)
    range_2 = random.uniform(5, 10)
    range_3 = random.uniform(10, 25)
    range_4 = random.uniform(25, 50)
    range_5 = random.uniform(50, 100)

    range_l = random.randint(0, 20)
'''





