import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import minimize, brute
from scipy.stats import chi2
#LECs = np.loadtxt('/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/NNs/LEC_ratio/LEC_files/phaseshifts_SLLJT_10010_lambda_2.00_s5_new_2.dat')[:240, 0:5]
energy_lin = np.linspace(0, 200, 200)
print(energy_lin[100])

phaseshifts = np.loadtxt('/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/NNs/LEC_ratio/LEC_files/phaseshifts_SLLJT_10010_lambda_2.00_s5_new_2.dat')[:240,5: 205]
key_phaseshifts = np.array([phaseshifts[:, 10], phaseshifts[:, 20], phaseshifts[:, 50], phaseshifts[:, 100], phaseshifts[:, 175]])
key_phaseshifts = np.array([phaseshifts[:, 2], phaseshifts[:, 10], phaseshifts[:, 20], phaseshifts[:, 50], phaseshifts[:, 100]])

phaseshift_tiles=[ r"$\delta(10\,$MeV)", r"$\delta(20\,$MeV)", r"$\delta(50\,$MeV)", r"$\delta(100\,$MeV)", r"$\delta(175\,$MeV)"]
phaseshift_tiles=[ r"$\delta(2\,$MeV)", r"$\delta(10\,$MeV)", r"$\delta(20\,$MeV)", r"$\delta(50\,$MeV)", r"$\delta(100\,$MeV)"]

LECs = key_phaseshifts
weights = np.loadtxt('/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/advanced_weights/weights_pdf_custom_grid.txt')

rgba_color = (203 / 255, 139 / 255, 136 / 255, 255 / 255)

rgba2 = (136/255, 160/255, 203/255, 1)

rgba3 = (121/255, 192/255, 116/255, 1)

fig, axes = plt.subplots(5, 5, figsize=(12, 12))#, sharex='col', sharey='row')
'''
for i in range(5):
    for j in range(5):

        if j <= i:



            axes[i, j].scatter(LECs[:,i], LECs[:,j], color=rgba_color, s=5, alpha =(weights/max(weights))**2)
            #axes[i, j].set_title(f'Plot {i+1}-{j+1}')
        else:
            axes[i, j].axis('off')
'''
#variances = [0.14237304, 0.10260972, 0.16037478, 1.05816881, 6.11110228]





for i in range(5):
    for j in range(5):

        if j <= i:


            ps_i, ps_j = key_phaseshifts[i,:], key_phaseshifts[j,:]
            axes[i, j].scatter(ps_i, ps_j, color=rgba_color, s=5, alpha =(weights/max(weights)))
            data = np.array([ ps_i, ps_j])




            
            #scaled_data  = np.array([ ps_i* weights, ps_j* weights]) bs, changes values not weights
            #mean_vector = np.mean(data, axis=1)
            mean_vector = np.average(data, axis=1, weights=weights/max(weights))
            # Step 2: Compute the covariance matrix
            covariance_matrix = np.cov(data, rowvar=1, aweights=weights/max(weights))
            #covariance_matrix = np.array([[variances[i], 0], [0, variances[j]]])
            axes[i,j].scatter(mean_vector[0], mean_vector[1], color=rgba2, marker='p', label='Mean')
            #axes[i, j].set_title(f'Plot {i+1}-{j+1}')
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
            width1, height1 = 2 * np.sqrt(2.278  * eigenvalues)  #68%, 5.991 for  95% confidence ellipse (2.447 is the chi-square value for 2 degrees of freedom)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            ellipse1 = plt.matplotlib.patches.Ellipse(xy=mean_vector, width=width1, height=height1, angle=angle,
                                                     edgecolor=rgba2, facecolor='none', lw=2, linestyle='dashed')
            e1_plot = axes[i, j].add_patch(ellipse1)

            width2, height2 = 2 * np.sqrt(5.991 * eigenvalues)
            ellipse2 = plt.matplotlib.patches.Ellipse(xy=mean_vector, width=width2, height=height2, angle=angle,
                                                     edgecolor=rgba3, facecolor='none', lw=2, linestyle='dashed')
            e2_plot = axes[i, j].add_patch(ellipse2)
            #axes[i, j].set_title(f'Plot {phaseshift_tiles[i]}-{phaseshift_tiles[j]}')
            axes[i, j].set_xlabel(phaseshift_tiles[i])
            axes[i, j].set_ylabel(phaseshift_tiles[j])
        else:
            axes[i, j].axis('off')
legend_handles = [Line2D([0], [0], marker='o', color=rgba2, label='68%', markerfacecolor=rgba2, markersize=10),
                  Line2D([0], [0], marker='s', color=rgba3, label='95%', markerfacecolor=rgba3, markersize=10)]

# Add a legend to the plot
fig.legend(handles=legend_handles, loc='center')
#fig.legend(e1_plot, '68%')
plt.tight_layout()
fig.suptitle('phase shifts at different energies correlation')
plt.savefig('phase_shift_correlations.pdf')
plt.show()