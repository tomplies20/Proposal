import numpy as np
import matplotlib.pyplot as plt



def triton(i):
    plt.subplot(2, 3, i)

    # Sample data
    data = [3, 6, 8, 10, 12, 16, 18, 20, 23, 26, 28, 30]

    # Define the range size
    x = 5

    # Create bins for grouping values within the range x
    bins = np.arange(0, max(data) + x, x)

    # Group the data into bins
    hist, _ = np.histogram(data, bins)

    # Plot the bar chart
    plt.bar(bins[:-1], hist, width=x, align='edge')

    # Customize labels and title
    plt.xlabel(r'E$_{\mathrm{Triton}}$ (MeV)')
    #plt.ylabel('Frequency')
    #plt.title('Bar Plot with Grouped Values within Range')

    # Show the plot
    plt.show()

def LEC(i):
    plt.subplot(2, 3, i)





plt.figure(figsize = (12, 8))
triton(1)

plt.tight_layout()

plt.show()
