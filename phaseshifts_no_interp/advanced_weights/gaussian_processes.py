from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from sklearn.gaussian_process.kernels import RBF
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.stats import multivariate_normal


import pymc as pm
from pymc.gp.cov import *
from pymc.gp.util import stabilize

#import tensorflow as tf


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
#energy = np.linspace(energy_[0], 200 + energy_[0], grid_size )
grid_size = 200
upper_energy_limit = 200
energy = np.linspace((energy_[0]/upper_energy_limit)**(1/3), 1, grid_size)**3 * upper_energy_limit


#parameters

lin_size = 200


l = 40

new_upper_limit = 200



energy_lin = np.linspace(energy[0], new_upper_limit, lin_size)
#energy_lin = np.linspace(energy_[0], 200, 200)
print(len(energy))

energy_lin_200 = np.linspace(energy[0], 200, 200)


partial_wave = '10010'


data_ = np.loadtxt(f'/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/random_sampling/phaseshifts_SLLJT_{partial_wave}_lambda_2.00_s5_new_2.dat')
data_ = np.loadtxt('/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/NNs/LEC_ratio/LEC_files/phaseshifts_SLLJT_10010_lambda_2.00_s5_new_2.dat')
#data_ = np.loadtxt('/Users/pleazy/Desktop/phaseshifts_SLLJT_01110_lambda_2.00_s5.dat')

LECs = data_[:, 0:5]
#ratios = data_[:,5]
phaseshifts = data_[:,5:205]
print(np.shape(phaseshifts))
phaseshifts_interp = np.zeros((len(phaseshifts[:, 0]), lin_size))
for k in range(len(phaseshifts[:, 0])):
    ps_interpolate = sc.interpolate.interp1d(energy_lin_200, phaseshifts[k, :])
    phaseshifts_interp[k] = ps_interpolate(energy_lin)

#phaseshifts_interp = phaseshifts

def gaussian(x, mu, sigma):
    return 1/(np.sqrt( 2 * np.pi) * sigma) * np.exp( -1/2 * (( x - mu )/sigma)**2)



errors = np.loadtxt(
    f'/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/phaseshift_files/EKM_uncertainty/phaseshifts_uncertainties_SLLJT_{partial_wave}_lambda_2.00_s5.dat')

N3LO_error = errors[:, 3]
error_interpolate = sc.interpolate.interp1d(energy_, N3LO_error)
error_interpolated = error_interpolate(energy_lin)

reference_data = np.loadtxt(
    f'/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/phaseshift_files/phaseshifts_SVD/phaseshifts_unchanged_SLLJT_{partial_wave}_lambda_2.00_s5.dat')
reference = reference_data[:, 3]
reference_interpolate = sc.interpolate.interp1d(energy_, reference)
reference_interpolated = reference_interpolate(energy_lin)


covariance_matrix = np.zeros((lin_size, lin_size))

signal_variance = np.abs(error_interpolated)

variances = signal_variance

mean = reference_interpolated
# Sample data (replace with your actual data)
#X = np.random.rand(100, 1)
#y = np.sin(X).ravel()
#variances = np.random.uniform(0.1, 0.5, len(X))

# Define a custom kernel to incorporate variances
class HeteroscedasticKernel(Kernel):
    def __init__(self, variances, base_kernel):
        self.variances = variances
        self.base_kernel = base_kernel

    def diag(self, X):
        # Diagonal of the kernel matrix
        return self.base_kernel.diag(X) + self.variances

    def is_stationary(self):
        # Check if the kernel is stationary
        return self.base_kernel.is_stationary()

    def __call__(self, X, Y=None, eval_gradient=False):
        if eval_gradient:
            raise ValueError("Gradient evaluation is not supported.")
        K = self.base_kernel(X, Y)
        K += np.diag(self.variances)
        return K

# Choose a base kernel (e.g., RBF)
base_kernel = 1.0 * RBF(length_scale=20.0)

# Create the custom kernel
heteroscedastic_kernel = HeteroscedasticKernel(variances, base_kernel)

# Initialize and fit the Gaussian process
gp = GaussianProcessRegressor(kernel=heteroscedastic_kernel)


X = energy_lin
X = X.reshape(-1, 1)
y = mean
y = phaseshifts_interp[2,:]
print('d')
print(np.shape(y))
gp.fit(X, y)




X_new = np.linspace(0, 200, 200).reshape(-1, 1)
y_pred_mean, y_pred_std = gp.predict(X_new, return_std=True)
print(y_pred_std)
print(variances)
# Plot the results
plt.figure()
plt.scatter(X, y, c='k', label='Training Data')  # Plot the training data
plt.plot(X_new, y_pred_mean, 'r', lw=2, label='Mean Prediction')  # Plot the mean prediction
plt.fill_between(X_new[:, 0], y_pred_mean - 1.96 * y_pred_std, y_pred_mean + 1.96 * y_pred_std, alpha=0.2, color='r', label='95% Confidence Interval')  # Plot the confidence interval
plt.xlabel('X')
plt.ylabel('y')
plt.title('Gaussian Process Regression')
plt.legend(loc='upper left')
plt.show()