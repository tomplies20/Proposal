import numpy as np
import matplotlib.pyplot as plt
from phaseshift_calculator_LECs import *
import scipy as sc
from scipy.stats import multivariate_normal

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import minimize

energy_ = np.loadtxt('/Users/pleazy/PycharmProjects/Proposal/energy_.txt')

energy_lin = np.linspace(energy_[0], 200, 200)



energy_lin = np.array([1, 10, 20, 50, 100])

grid_size = len(energy_lin)

partial_wave = '10010'

data_points = 480
data = np.loadtxt('/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/NNs/LEC_ratio/LEC_files/phaseshifts_SLLJT_10010_lambda_2.00_s5_new_x.dat')[:480, :] #[:240, 0:5]
LECs = data[:, 0:5]
phaseshifts = data[:, 5: 205]


errors = np.loadtxt(
    f'/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/phaseshift_files/EKM_uncertainty/phaseshifts_uncertainties_SLLJT_{partial_wave}_lambda_2.00_s5.dat')

N3LO_error = errors[:, 3]
error_interpolate = sc.interpolate.interp1d(energy_, N3LO_error)
error_interpolated = error_interpolate(energy_lin)



phaseshifts_LO_raw = SVD_fixed_LECs(f'{partial_wave}', 0, 4, '2.00')
phaseshifts_NLO_raw = SVD_fixed_LECs(f'{partial_wave}', 1, 4, '2.00')
phaseshifts_N2LO_raw = SVD_fixed_LECs(f'{partial_wave}', 2, 4, '2.00')
phaseshifts_N3LO_raw = SVD_fixed_LECs(f'{partial_wave}', 3, 4, '2.00')
phaseshifts_N4LO_raw = SVD_fixed_LECs(f'{partial_wave}', 4, 4, '2.00')


LO_interpolate = sc.interpolate.interp1d(energy_, phaseshifts_LO_raw)
phaseshifts_LO = LO_interpolate(energy_lin)


NLO_interpolate = sc.interpolate.interp1d(energy_, phaseshifts_NLO_raw)
phaseshifts_NLO = NLO_interpolate(energy_lin)


N2LO_interpolate = sc.interpolate.interp1d(energy_, phaseshifts_N2LO_raw)
phaseshifts_N2LO = N2LO_interpolate(energy_lin)


N3LO_interpolate = sc.interpolate.interp1d(energy_, phaseshifts_N3LO_raw)
phaseshifts_N3LO = N3LO_interpolate(energy_lin)


N4LO_interpolate = sc.interpolate.interp1d(energy_, phaseshifts_N4LO_raw)
phaseshifts_N4LO = N4LO_interpolate(energy_lin)

reference_phaseshift = phaseshifts_N4LO

c_LO =  phaseshifts_LO
c_NLO = (phaseshifts_NLO - phaseshifts_LO) / reference_phaseshift
c_N2LO = (phaseshifts_N2LO - phaseshifts_NLO) / reference_phaseshift
c_N3LO = (phaseshifts_N3LO - phaseshifts_N2LO) / reference_phaseshift
c_N4LO = (phaseshifts_N4LO - phaseshifts_N3LO) / reference_phaseshift




'''
Q = np.empty([len(energy)])
for k in range(len(energy)):
    if energy[k] < 140:
        Q[k] = np.sqrt(M*140/(2*200*200))
    else:
        Q[k] = np.sqrt(M*energy_lin[k]/(2*200*200))
'''
def Q_function(energies):
    Q = np.zeros((len(energies)))
    for i, energy in enumerate(energies):
        if energy < 140:
            Q[i] = np.sqrt(M * 140 / (2 * 200 * 200))
        else:
            Q[i] = np.sqrt(M*energy/(2*200*200))
    return Q




def M_N3LO(m, energy):
    return reference_phaseshift * Q_function(energy)**4 / (1-Q_function(energy)) * m





def K_N3LO( l):
    reference_phaseshift * reference_phaseshift * (Q[i] * Q[j]) ** 4 / (1 - Q[i] * Q[j]) * kernel(i, j, l)



class CustomMean:
    def __init__(self, mean_param):
        self.mean_param = mean_param


    def __call__(self, X):
        QX = Q_function(X)

        refX = reference_phaseshift
        return refX * QX**4 / (1- QX) * self.mean_param

class CustomKernel(RBF):
    def __init__(self, length_scale, variance_scale):
        super(CustomKernel, self).__init__(length_scale)
        self.variance_scale = variance_scale
    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X

        # Calculate the Q values for input data points X and Y
        QX = Q_function(X)  # Define `params` for Q function
        QY = Q_function(X)

        refX = reference_phaseshift
        refY = reference_phaseshift
        # Calculate the custom kernel
        K = refX * refY * (QX * QY)**4 / (1 - QX * QY) * super(CustomKernel, self).__call__(X, Y, eval_gradient=False) * self.variance_scale
        '''
        if eval_gradient:
            # Gradient calculations can be added if needed
            raise NotImplementedError("Gradient not implemented for this custom kernel")
        '''
        if eval_gradient:
            # If you don't want to compute gradients, return None
            return K, None
        else:
            return K


def negative_log_likelihood(params, X, y):
    # Extract hyperparameters
    length_scale = params[0]
    #mean_param = params[1]
    variance_scale = params[2]

    # Create a custom kernel with the hyperparameters
    custom_kernel = CustomKernel(length_scale, variance_scale)

    # Create the Gaussian Process model
    gp = GaussianProcessRegressor(kernel=custom_kernel, alpha=1e-4)

    # Fit the GP model to the data
    gp.fit(X, y)

    # Compute the negative log likelihood (change as needed based on your likelihood model)
    log_likelihood = gp.log_marginal_likelihood()

    # Return the negative log likelihood (to be minimized)
    return -log_likelihood

params = [1, 0, 1]


X = energy_lin.reshape(-1, 1)
y = c_N3LO

# Optimize the hyperparameters
result = minimize(negative_log_likelihood, params, args=(X, y), method='L-BFGS-B')
optimal_params = result.x

'''
mean = np.zeros((grid_size))
cov = np.zeros((grid_size, grid_size))

for k in range(grid_size):
    mean[k] = M_N3LO(m, k)


for i in range(grid_size):
    for j in range(grid_size):
        cov[i, j] = K_N3LO(i, j, l)



mvn = multivariate_normal(mean= mean, cov=cov)
'''