import numpy as np
import matplotlib.pyplot as plt

triton_data = np.loadtxt('triton_results.txt')

samples_path = '/Users/pleazy/PycharmProjects/Proposal/partial_waves_preparation/samples_after_sampling_partial_waves/'
data_00001 = np.loadtxt(samples_path + 'samples_SLLJT_00001_lambda_2.00_s5.dat')[:len(triton_data), 0:5]
data_10010 = np.loadtxt(samples_path + 'samples_SLLJT_10010_lambda_2.00_s5.dat')[:len(triton_data), 0:5]
data_01110 = np.loadtxt(samples_path + 'samples_SLLJT_01110_lambda_2.00_s5.dat')[:len(triton_data), 0:5]
data_11101 = np.loadtxt(samples_path + 'samples_SLLJT_11101_lambda_2.00_s5.dat')[:len(triton_data), 0:5]
data_11111 = np.loadtxt(samples_path + 'samples_SLLJT_11111_lambda_2.00_s5.dat')[:len(triton_data), 0:5]
data_11121 = np.loadtxt(samples_path + 'samples_SLLJT_11121_lambda_2.00_s5.dat')[:len(triton_data), 0:5]

partial_waves_indices = {
    "00001": data_00001,
    "10010": data_10010,
    "01110": data_01110,
    "11101": data_11101,
    "11111": data_11111,
    "11121": data_11121
}

sv_names = [r's$_{1}$', r's$_{2}$', r's$_{3}$', r's$_{4}$', r's$_{5}$']

custom_color = (0.7, 0.2, 0.2)
rgba_color = (203 / 255, 139 / 255, 136 / 255, 255 / 255)

def triton(row):
    plt.subplot(6, 6, row*6 +1)

    # Sample data
    #data = [3, 6, 8, 10, 12, 16, 18, 20, 23, 26, 28, 30]

    # Define the range size
    x = 0.1

    # Create bins for grouping values within the range x
    bins_ = np.arange(min(triton_data) , min(triton_data) +2  , x)
    bins = bins_[:-1]

    # Group the data into bins
    hist, _ = np.histogram(triton_data, bins)

    # Plot the bar chart
    #plt.bar(bins[:-1], hist, width=x, align='edge')
    plt.bar(bins[:-1], hist, width=x, align='edge', color=rgba_color)

    # Customize labels and title
    plt.xlabel(r'E$_{\mathrm{Triton}}$ (MeV)')
    #plt.ylabel('Frequency')
    #plt.title('Bar Plot with Grouped Values within Range')

    # Show the plot


def LEC(partial_wave, row):
    LECs = partial_waves_indices[partial_wave]
    for i in range(0, 5):

        plt.subplot(6, 6, row*6 + 2 + i)


        sv = LECs[:,i]
        #plt.title(sv_names[i])
        plt.xlabel(sv_names[i])
        plt.scatter(sv, triton_data, color = rgba_color)







plt.figure(figsize = (30, 30))
triton(0)
LEC('00001', 0)
triton(1)
LEC('10010', 1)
triton(2)
LEC('01110', 2)
triton(3)
LEC('11101', 3)
triton(4)
LEC('11111', 4)
triton(5)
LEC('11121', 5)

plt.tight_layout()
plt.savefig('sensitivity_analysis.pdf')
plt.show()
