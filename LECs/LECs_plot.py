import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('/Users/pleazy/PycharmProjects/Proposal/phaseshifts/random_sampling/phaseshifts_SLLJT_00001_lambda_2.00_s5_15000.dat')
LECs = data[:, 0:5]



fig, axes = plt.subplots(5, 5, figsize=(12, 12))#, sharex='col', sharey='row')



for i in range(5):
    for j in range(5):
        # Only plot if we are in the left half (including the diagonal)
        if j <= i:


            # Plot the data on the current subplot
            axes[i, j].scatter(LECs[:,i], LECs[:,j], s=0.1)
            #axes[i, j].set_title(f'Plot {i+1}-{j+1}')
        else:
            axes[i, j].axis('off')

#for i, label in enumerate([r'$s_1$', r'$s_2$', r'$s_3$', r'$s_4$', r'$s_5$']):
#    axes[i, 0].annotate(label, xy=(0.5, 0.5), xytext=(-axes[i, 0].yaxis.labelpad - 5, 0),
#                        textcoords='offset points', ha='center', va='center', rotation=90)
#    axes[4, i].annotate(label, xy=(0.5, 0.5), xytext=(0, axes[4, i].xaxis.labelpad + 10),
#                        textcoords='offset points', ha='center', va='center')

# Adjust layout to prevent subplot overlap
#plt.tight_layout()

# Show the plots
plt.savefig('LECS_random_sampling.pdf')
plt.show()
