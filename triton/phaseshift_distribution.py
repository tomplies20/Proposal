import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy as sc
data = np.loadtxt('/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/NNs/LEC_ratio/LEC_files/phaseshifts_SLLJT_10010_lambda_2.00_s5_new_x.dat')

partial_wave = '10010'
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
#for e in energy_:
#    print(e)
#energy = np.linspace(energy_[0], 200 + energy_[0], grid_size )
grid_size = 200
upper_energy_limit = 200

up_lim = 2

energy = np.linspace((energy_[0]/upper_energy_limit)**(1/3), 1, grid_size)**3 * upper_energy_limit
#energy_lin = np.linspace(energy[0], 200, 200)
energy_lin_200 = np.linspace(energy[0], 200, 200)
energy_lin = np.linspace(energy[0], up_lim, 200)
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

def gaussian(x,A, mu, sigma):
#    return 1/(np.sqrt( 2 * np.pi) * sigma) * np.exp( -1/2 * (( x - mu )/sigma)**2)
    return A*np.exp( -1/2 * (( x - mu )/sigma)**2)


LECs = data[:, 0:5]
phaseshifts = data[:, 5:205]
for i, phaseshift_ in enumerate(phaseshifts):
    phaseshifts_interpolate = sc.interpolate.interp1d(energy_lin_200, phaseshift_)
    phaseshifts_interpolated = phaseshifts_interpolate(energy_lin)
    phaseshifts[i] = np.copy(phaseshifts_interpolated)
    #phaseshift_ = phaseshifts_interpolated
#print(np.shape(phaseshifts))

#weights_ = np.ones((240))

reverser = np.linspace(1,0, 200)

#weights_ = data[:, 205]
#weights_ = weights_ /reverser
#np.shape(weights_)

energy_index = 20

weights_ = np.loadtxt('/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/advanced_weights/weights_pdf_custom_grid_x.txt')
#print(np.shape(weights_))
phaseshifts_at_one_energy = phaseshifts[:, energy_index]
print(np.shape(phaseshifts_at_one_energy))
hist, bin_edges = np.histogram(phaseshifts_at_one_energy, bins=20, weights=weights_)

plt.hist(phaseshifts_at_one_energy, bins=bin_edges, weights=weights_, edgecolor='black', alpha=0.7)




# Combine data and weights into tuples
data_weight_pairs = list(zip(phaseshifts_at_one_energy, weights_))

# Sort the data based on values while preserving associated weights
sorted_data_weight_pairs = sorted(data_weight_pairs, key=lambda x: x[0])

# Calculate the total weight
total_weight = sum(weight for _, weight in sorted_data_weight_pairs)

# Calculate indices for weighted percentiles
target_percentiles = [16, 50, 84]
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
#print("Weighted 25th Percentile:", percentiles[25])
#print("Weighted 75th Percentile:", percentiles[75])




percentile_16_bin_edge = percentiles[16]
percentile_50_bin_edge = percentiles[50]
percentile_84_bin_edge = percentiles[84]


plt.vlines(percentile_16_bin_edge, 0, 15, color='r')
plt.vlines(percentile_50_bin_edge, 0, 15, color='g')
plt.vlines(percentile_84_bin_edge, 0, 15, color='r')
# Print the results
print("25th Percentile:", percentile_16_bin_edge)
print("Median:", percentile_50_bin_edge)
print("75th Percentile:", percentile_84_bin_edge)


### gaussian fit

'''

bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
params, _ = curve_fit(gaussian, bin_centers, hist, p0=[1, phaseshifts_at_one_energy[50], 0.2])
x = np.linspace(min(phaseshifts_at_one_energy), max(phaseshifts_at_one_energy), 100)
fitted_curve = gaussian(x, *params)
plt.plot(x, fitted_curve, 'r', label='Fitted Gaussian')



plt.vlines(reference_interpolated[energy_index],0, max(hist*1.2), color='black', label='correct phaseshift')
plt.vlines(params[1]+params[2], 0, max(hist*1.2), color='red', label=r'1$\sigma$ fit')
plt.vlines(params[1]-params[2], 0, max(hist*1.2), color='red')

'''


plt.vlines(reference_interpolated[energy_index],0, max(hist*1.2), color='black', label='correct phaseshift')
plt.vlines(reference_interpolated[energy_index] + error_interpolated[energy_index], 0, max(hist*1.2), label=r'1$\sigma$ (EKM)')
plt.vlines(reference_interpolated[energy_index] - error_interpolated[energy_index], 0, max(hist*1.2))




#plt.vlines(error_interpolated[energy_index])

plt.title(fr"{np.round(energy_lin[energy_index], 2)}MeV")

plt.ylim(0, max(hist*1.2))
#plt.plot()
plt.show()
print(np.shape(phaseshifts))

'''
fig, axes = plt.subplots(5, 5, figsize=(12, 12))#, sharex='col', sharey='row')

weights = data[:, 205]

for i in range(5):
    for j in range(5):
        # Only plot if we are in the left half (including the diagonal)
        if j <= i:
            for m in range(len(LECs[:,0])):


                # Plot the data on the current subplot
                axes[i, j].scatter(LECs[:,i], LECs[:,j], s=(weights/max(weights))**2)
            #axes[i, j].set_title(f'Plot {i+1}-{j+1}')
                else:
                    axes[i, j].axis('off')
#plt.savefig('LEC_correlations_11111_68_percent.pdf')
plt.show()
'''
rgba_color = (203 / 255, 139 / 255, 136 / 255, 255 / 255)
weights_ = weights_/np.max(weights_)


indices = [13, 20, 24, 37, 66, 155, 201, 237, 253, 260, 264, 277, 306, 395, 441, 477]
for i, phaseshifts_ in enumerate(phaseshifts[:, :]): #:240 prev
    plt.plot(energy_lin, phaseshifts_ / reference_interpolated, color='grey', linewidth=0.5, alpha=weights_[i])
    for j in indices:
        if i ==j:
            plt.plot(energy_lin, phaseshifts_ / reference_interpolated, color='r', linewidth=0.5)#, alpha=weights_[i])

plt.fill_between(energy_lin, (reference_interpolated - error_interpolated)/reference_interpolated, (reference_interpolated + error_interpolated)/reference_interpolated, color='orange', alpha=0.4)
plt.plot(energy_lin, (reference_interpolated - error_interpolated)/reference_interpolated, linestyle='dashed', color='orange', alpha = 0.6)
plt.plot(energy_lin, (reference_interpolated + error_interpolated)/reference_interpolated, linestyle='dashed', color='orange', alpha = 0.6)
plt.ylim(0.8, 1.1)
plt.xlim(0, up_lim)
#plt.savefig('phaseshift_distribution_after_weights_overbind_small_grid.pdf')
plt.show()

'''
for i, phaseshifts_ in enumerate(phaseshifts[:, :]): #:240 prev
    plt.plot(energy_lin, phaseshifts_ , color='grey', linewidth=0.5, alpha=weights_[i])
    for j in indices:
        if i ==j:
            plt.plot(energy_lin, phaseshifts_ , color='r', linewidth=0.5, alpha=weights_[i])
plt.show()
'''