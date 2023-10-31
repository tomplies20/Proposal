import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

triton_data = np.loadtxt('triton_newest.txt')

#samples_path = '/Users/pleazy/PycharmProjects/Proposal/partial_waves_preparation/samples_after_sampling_partial_waves/'

#data_10010 = np.loadtxt(samples_path + 'samples_SLLJT_10010_lambda_2.00_s5.dat')[:len(triton_data), 0:5]

#weights = np.loadtxt(samples_path + 'samples_SLLJT_10010_lambda_2.00_s5.dat')[:len(triton_data), 205]
data_10010 = np.loadtxt('/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/NNs/LEC_ratio/LEC_files/phaseshifts_SLLJT_10010_lambda_2.00_s5_new_2.dat')[:len(triton_data), 0:5]

weights = np.loadtxt('/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/NNs/LEC_ratio/LEC_files/phaseshifts_SLLJT_10010_lambda_2.00_s5_new_2.dat')[:len(triton_data), 205]
weights = np.loadtxt('/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/NNs/LEC_ratio/LEC_files/new_weights_2.dat')


####  multivariate gauss weights
weights = np.loadtxt('/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/advanced_weights/weights_pdf.txt')


partial_waves_indices = {
    "10010": data_10010,

}

sv_names = [r's$_{1}$', r's$_{2}$', r's$_{3}$', r's$_{4}$', r's$_{5}$']

custom_color = (0.7, 0.2, 0.2)
rgba_color = (203 / 255, 139 / 255, 136 / 255, 255 / 255)
'''
temp = weights
weights = temp / np.max(temp)
'''
print('effective number of samples: ' + str(np.sum(weights)))
def gaussian(x,A, mu, sigma):
    #return 1/(np.sqrt( 2 * np.pi) * sigma) * np.exp( -1/2 * (( x - mu )/sigma)**2)
    return A*np.exp( -1/2 * (( x - mu )/sigma)**2)






def triton(row, weights, number_bins):
    #plt.subplot(1,6, row*6 +1)

    '''
    # Define the range size
    x = 0.1

    # Create bins for grouping values within the range x
    bins_ = np.arange(min(triton_data) , min(triton_data) +2  , x)
    bins = bins_[:-1]

    # Group the data into bins
    hist, _ = np.histogram(triton_data , bins)

    # Plot the bar chart
    #plt.bar(bins[:-1], hist, width=x, align='edge')
    plt.bar(bins[:-1], hist, width=x, align='edge', color=rgba_color)
    '''


    #weights = np.ones((len(triton_data)))
    weights_ = weights**1
    hist, bin_edges = np.histogram(triton_data, bins=number_bins, weights=weights_)
    # Plot the weighted histogram
    #plt.hist(triton_data, bins=bin_edges, weights=weights_, edgecolor='black', alpha=0.7)

    # Customize labels and title
    #plt.xlabel(r'E$_{\mathrm{Triton}}$ (MeV)')
    #plt.ylabel('Frequency')
    #plt.title('Bar Plot with Grouped Values within Range')

    # Show the plot

    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    params, _ = curve_fit(gaussian, bin_centers, hist, p0=[1, -8.4, 0.2])
    x = np.linspace(min(triton_data), max(triton_data), 100)
    #fitted_curve = gaussian(x, *params)
    #plt.plot(x, fitted_curve, label='Fitted Gaussian', color = rgba_color)

    #-8.445221, 1, xerr=0.04114279080829548
    #plt.vlines(params[1] + 0.04114279080829548, 0, 15, color='black')
    #plt.vlines(params[1] - 0.04114279080829548, 0, 15, color='black')
    #plt.ylim(0, max(hist)*1.1)
    return params[2]
'''
histogram_weights = weights*8
for m in range(len(histogram_weights)):
    if histogram_weights[m] < 0.0000001:
        histogram_weights[m] = 0
    if histogram_weights[m] > 1:
        histogram_weights[m] = 1

def LEC(partial_wave, row):
    LECs = partial_waves_indices[partial_wave]
    for i in range(0, 5):

        plt.subplot(1,6, row*6 + 2 + i)


        sv = LECs[:,i]
        #plt.title(sv_names[i])
        plt.xlabel(sv_names[i])



        plt.scatter(sv, triton_data, color = rgba_color, alpha = histogram_weights)







plt.figure(figsize = (30, 5))
triton(0)
LEC('10010', 0)


plt.tight_layout()
plt.savefig('sensitivity_analysis_3s1_new.pdf')
plt.show()

weights_ = weights ** 0
hist, bin_edges = np.histogram(weights, bins=14, weights=weights_)
# Plot the weighted histogram
plt.hist(weights, bins=bin_edges, weights=weights_, edgecolor='black', alpha=0.7)
plt.savefig('weights_triton_newest.pdf')
plt.show()


#log_bins = np.logspace(np.log10(weights.min()), np.log10(weights.max()), 20)
#hist, bins = np.histogram(weights, bins=log_bins)
#plt.xscale('log')
#plt.xlim(0, 0.0001)
# Plot the logarithmic histogram

#plt.hist(weights, bins=log_bins, alpha=0.7, color='blue', edgecolor='black', linewidth=1.2)
#plt.show()

'''




def triton_plot(row, weights, number_bins):
    #plt.subplot(1,6, row*6 +1)



    #weights = np.ones((len(triton_data)))
    weights_ = weights**1
    hist, bin_edges = np.histogram(triton_data, bins=number_bins, weights=weights_)
    # Plot the weighted histogram
    plt.hist(triton_data, bins=bin_edges, weights=weights_, edgecolor='black', alpha=0.7)

    # Customize labels and title
    plt.xlabel(r'E$_{\mathrm{Triton}}$ (MeV)')
    #plt.ylabel('Frequency')
    #plt.title('Bar Plot with Grouped Values within Range')

    # Show the plot

    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    params, _ = curve_fit(gaussian, bin_centers, hist, p0=[1, -8.4, 0.2])
    x = np.linspace(min(triton_data), max(triton_data), 100)
    fitted_curve = gaussian(x, *params)
    plt.plot(x, fitted_curve, label='Fitted Gaussian', color = rgba_color)

    #-8.445221, 1, xerr=0.04114279080829548
    plt.vlines(params[1] + 0.04114279080829548, 0, 15, color='black')
    plt.vlines(params[1] - 0.04114279080829548, 0, 15, color='black')
    plt.ylim(0, max(hist)*1.1)
    #return params[2]
    plt.tight_layout()
    # plt.savefig('sensitivity_analysis_3s1_new.pdf')
    plt.show()




#plt.figure(figsize = (30, 5))
#triton(0)
#LEC('10010', 0)




new_size = 100

number_bins = 10

standard_deviations = np.zeros(new_size)

for i in range(new_size):
    standard_deviations[i] = triton(0, weights[:, i], number_bins)
    print(str(i) + r' sigma$: ' + str(standard_deviations[i]))
x = np.linspace(0, new_size, new_size)
plt.scatter(x, standard_deviations, label='generated from covariance matrix without correlation')
plt.ylabel('standard deviation triton')
plt.xlabel('number of grid points / dimension likelihood function')
plt.legend()
plt.savefig('standard_deviation_convergence.pdf')
plt.show()

triton_plot(0, weights[:, 30], number_bins)