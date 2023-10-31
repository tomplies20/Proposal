import numpy as np
import matplotlib.pyplot as plt
from phaseshift_calculator_LECs import *
import scipy as sc
from scipy.stats import multivariate_normal


from scipy.optimize import minimize
from scipy.optimize import Bounds
from pyDOE import lhs
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

#%matplotlib inline



energy_ = np.loadtxt('/Users/pleazy/PycharmProjects/Proposal/energy_.txt')

energy_lin = np.linspace(energy_[0], 200, 200)



energy_lin = np.array([2, 10, 20, 50, 100])

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

training_points_LO =  phaseshifts_LO
training_points_NLO = (phaseshifts_NLO - phaseshifts_LO) / reference_phaseshift
training_points_N2LO = (phaseshifts_N2LO - phaseshifts_NLO) / reference_phaseshift
training_points_N3LO = (phaseshifts_N3LO - phaseshifts_N2LO) / reference_phaseshift
training_points_N4LO = (phaseshifts_N4LO - phaseshifts_N3LO) / reference_phaseshift




'''
Q = np.empty([len(energy)])
for k in range(len(energy)):
    if energy[k] < 140:
        Q[k] = np.sqrt(M*140/(2*200*200))
    else:
        Q[k] = np.sqrt(M*energy_lin[k]/(2*200*200))

'''

def M_N3LO(m, i):
    return reference_phaseshift * Q[i]**4 / (1-Q[i]) * m



'''
def kernel(i,j, l):
    distance = abs(energy_lin[i] - energy_lin[j])
    print('energy i: ' + str(energy_lin[i]))
    print('energy j: ' + str(energy_lin[j]))
    print('distance ' + str(distance))
    return np.exp(-distance ** 2 / (2 * l ** 2))

def K_N3LO(i, j, l, c):
    reference_phaseshift[i] * reference_phaseshift[j] * (Q[i] * Q[j]) ** 4 / (1 - Q[i] * Q[j]) * kernel(i, j, l) *c**2


mean = np.zeros((grid_size))
cov = np.zeros((grid_size, grid_size))

for k in range(grid_size):
    mean[k] = M_N3LO(m, k)


for i in range(grid_size):
    for j in range(grid_size):
        cov[i, j] = K_N3LO(i, j, l)


def gaussian(i, j, l, c, m):
    determinant = np.linalg.det()
    return (2*np.pi)** (-grid_size/2) * determinant ** (-1/2)
mvn = multivariate_normal(mean= mean, cov=cov)

'''


class GaussianProcess:
    """A Gaussian Process class for creating and exploiting
    a Gaussian Process model"""

    def __init__(self, n_restarts, optimizer):
        """Initialize a Gaussian Process model

        Input
        ------
        n_restarts: number of restarts of the local optimizer
        optimizer: algorithm of local optimization"""

        self.n_restarts = n_restarts
        self.optimizer = optimizer

    def Corr(self, X1, X2, theta):
        """Construct the correlation matrix between X1 and X2

        Input
        -----
        X1, X2: 2D arrays, (n_samples, n_features)
        theta: array, correlation legnths for different dimensions

        Output
        ------
        K: the correlation matrix
        """
        K = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            K[i, :] = np.exp(-np.sum(theta * (X1[i, :] - X2) ** 2, axis=1))

        return K

    def Neglikelihood(self, theta):
        """Negative likelihood function

        Input
        -----
        theta: array, logarithm of the correlation legnths for different dimensions

        Output
        ------
        LnLike: likelihood value"""

        theta = 10 ** theta  # Correlation length
        n = self.X.shape[0]  # Number of training instances
        one = np.ones((n, 1))  # Vector of ones

        # Construct correlation matrix
        K = self.Corr(self.X, self.X, theta) + np.eye(n) * 1e-10
        inv_K = np.linalg.inv(K)  # Inverse of correlation matrix

        # Mean estimation
        mu = (one.T @ inv_K @ self.y) / (one.T @ inv_K @ one)

        # Variance estimation
        SigmaSqr = (self.y - mu * one).T @ inv_K @ (self.y - mu * one) / n

        # Compute log-likelihood
        DetK = np.linalg.det(K)
        LnLike = -(n / 2) * np.log(SigmaSqr) - 0.5 * np.log(DetK)

        # Update attributes
        self.K, self.inv_K, self.mu, self.SigmaSqr = K, inv_K, mu, SigmaSqr

        return -LnLike.flatten()

    def fit(self, X, y):
        """GP model training

        Input
        -----
        X: 2D array of shape (n_samples, n_features)
        y: 2D array of shape (n_samples, 1)
        """

        self.X, self.y = X, y
        lb, ub = -3, 2

        # Generate random starting points (Latin Hypercube)
        lhd = lhs(self.X.shape[1], samples=self.n_restarts)

        # Scale random samples to the given bounds
        initial_points = (ub - lb) * lhd + lb

        # Create A Bounds instance for optimization
        bnds = Bounds(lb * np.ones(X.shape[1]), ub * np.ones(X.shape[1]))

        # Run local optimizer on all points
        opt_para = np.zeros((self.n_restarts, self.X.shape[1]))
        opt_func = np.zeros((self.n_restarts, 1))
        for i in range(self.n_restarts):
            res = minimize(self.Neglikelihood, initial_points[i, :], method=self.optimizer,
                           bounds=bnds)
            opt_para[i, :] = res.x
            opt_func[i, :] = res.fun

        # Locate the optimum results
        self.theta = opt_para[np.argmin(opt_func)]

        # Update attributes
        self.NegLnlike = self.Neglikelihood(self.theta)

    def predict(self, X_test):
        """GP model predicting

        Input
        -----
        X_test: test set, array of shape (n_samples, n_features)

        Output
        ------
        f: GP predictions
        SSqr: Prediction variances"""

        n = self.X.shape[0]
        one = np.ones((n, 1))

        # Construct correlation matrix between test and train data
        k = self.Corr(self.X, X_test, 10 ** self.theta)

        # Mean prediction
        f = self.mu + k.T @ self.inv_K @ (self.y - self.mu * one)

        # Variance prediction
        SSqr = self.SigmaSqr * (1 - np.diag(k.T @ self.inv_K @ k))

        return f.flatten(), SSqr.flatten()

    def score(self, X_test, y_test):
        """Calculate root mean squared error

        Input
        -----
        X_test: test set, array of shape (n_samples, n_features)
        y_test: test labels, array of shape (n_samples, )

        Output
        ------
        RMSE: the root mean square error"""

        y_pred, SSqr = self.predict(X_test)
        RMSE = np.sqrt(np.mean((y_pred - y_test) ** 2))

        return RMSE


def Test_1D(X):
    """1D Test Function"""

    y = (X * 6 - 2) ** 2 * np.sin(X * 12 - 4)

    return y


# Training data
#X_train = np.array([0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1], ndmin=2).T
X_train = energy_lin.reshape(-1, 1)
y_train = training_points_NLO.reshape(-1, 1)
#y_train = Test_1D(X_train)
# Testing data
X_test = np.linspace(0.0, 200, 100).reshape(-1, 1)
#y_test = Test_1D(X_test)

# GP model training
GP = GaussianProcess(n_restarts=10, optimizer='L-BFGS-B')
GP.fit(X_train, y_train)

# GP model predicting
y_pred, y_pred_SSqr = GP.predict(X_test)
sigma = y_pred_SSqr
plt.scatter(energy_lin, training_points_NLO)

plt.plot(X_test, y_pred, 'r', label='GP Prediction')
plt.fill_between(X_test[:, 0], y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, color='r', label='95% Confidence Interval')

plt.show()