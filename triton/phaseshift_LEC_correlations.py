import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import minimize, brute
from scipy.stats import chi2
energy_ = np.loadtxt('/Users/pleazy/PycharmProjects/Proposal/energy_.txt')
triton_data = np.loadtxt('triton_results_x.txt')[0:] #[0:240]

LECs = np.loadtxt('/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/NNs/LEC_ratio/LEC_files/phaseshifts_SLLJT_10010_lambda_2.00_s5_new_x.dat')[:, 0:5] #[:240, 0:5]
energy_lin = np.linspace(energy_[0], 200, 200)
print(energy_lin[100])

phaseshifts = np.loadtxt(
    '/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/NNs/LEC_ratio/LEC_files/phaseshifts_SLLJT_10010_lambda_2.00_s5_new_x.dat')[
              :, 5: 205]
key_phaseshifts = np.array(
    [phaseshifts[:, 10], phaseshifts[:, 20], phaseshifts[:, 50], phaseshifts[:, 100], phaseshifts[:, 175]])
key_phaseshifts = np.array(
    [phaseshifts[:, 2], phaseshifts[:, 10], phaseshifts[:, 20], phaseshifts[:, 50], phaseshifts[:, 100]])

key_phaseshifts = np.array(
    [phaseshifts[:, 1], phaseshifts[:, 10], phaseshifts[:, 20], phaseshifts[:, 50], phaseshifts[:, 100]])

phaseshift_tiles = [r"$\delta(10\,$MeV)", r"$\delta(20\,$MeV)", r"$\delta(50\,$MeV)", r"$\delta(100\,$MeV)",
                    r"$\delta(175\,$MeV)"]
phaseshift_tiles = [r"$\delta(1\,$MeV)", r"$\delta(10\,$MeV)", r"$\delta(20\,$MeV)", r"$\delta(50\,$MeV)",
                    r"$\delta(100\,$MeV)"]

LEC_titles = [r"$s_1$", r"$s_2$",  r"$s_3$", r"$s_4$", r"$s_5$"]
triton_title = [r"E$_{\text{Triton}}$"]

titles = np.append(phaseshift_tiles, LEC_titles)
titles = np.append(titles, triton_title)
print(np.shape(titles))

#LECs = key_phaseshifts
weights = np.loadtxt(
    '/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/advanced_weights/weights_pdf_custom_grid_x.txt')
#weights = np.ones((480))
rgba_color = (203 / 255, 139 / 255, 136 / 255, 255 / 255)

rgba2 = (136 / 255, 160 / 255, 203 / 255, 1)

rgba3 = (121 / 255, 192 / 255, 116 / 255, 1)

fig, axes = plt.subplots(11, 11, figsize=(24, 24))  # , sharex='col', sharey='row')

print(np.shape(key_phaseshifts))
print(np.shape(LECs))
ps_and_LECs = np.row_stack((key_phaseshifts, LECs.T))
ps_LECs_triton = np.row_stack((ps_and_LECs, triton_data))
print(np.shape(ps_and_LECs))

#weights = np.ones(240)

for i in range(11):
    for j in range(11):

        if j <= i:

            ps_i, ps_j = ps_LECs_triton[i, :], ps_LECs_triton[j, :]
            axes[i, j].scatter(ps_i, ps_j, color=rgba_color, s=5, alpha=(weights / max(weights)))
            data = np.array([ps_i, ps_j])

            # scaled_data  = np.array([ ps_i* weights, ps_j* weights]) bs, changes values not weights
            # mean_vector = np.mean(data, axis=1)
            mean_vector = np.average(data, axis=1, weights=weights / max(weights))
            # Step 2: Compute the covariance matrix
            covariance_matrix = np.cov(data, rowvar=1, aweights=weights / max(weights))
            # covariance_matrix = np.array([[variances[i], 0], [0, variances[j]]])
            axes[i, j].scatter(mean_vector[0], mean_vector[1], color=rgba2, marker='p', label='Mean')
            # axes[i, j].set_title(f'Plot {i+1}-{j+1}')
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
            width1, height1 = 2 * np.sqrt(
                2.278 * eigenvalues)  # 68%, 5.991 for  95% confidence ellipse (2.447 is the chi-square value for 2 degrees of freedom)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            ellipse1 = plt.matplotlib.patches.Ellipse(xy=mean_vector, width=width1, height=height1, angle=angle,
                                                      edgecolor=rgba2, facecolor='none', lw=2, linestyle='dashed')
            e1_plot = axes[i, j].add_patch(ellipse1)

            width2, height2 = 2 * np.sqrt(5.991 * eigenvalues)
            ellipse2 = plt.matplotlib.patches.Ellipse(xy=mean_vector, width=width2, height=height2, angle=angle,
                                                      edgecolor=rgba3, facecolor='none', lw=2, linestyle='dashed')
            e2_plot = axes[i, j].add_patch(ellipse2)
            # axes[i, j].set_title(f'Plot {phaseshift_tiles[i]}-{phaseshift_tiles[j]}')
            axes[i, j].set_xlabel(titles[i])
            axes[i, j].set_ylabel(titles[j])
        else:
            axes[i, j].axis('off')
legend_handles = [Line2D([0], [0], marker='o', color=rgba2, label='68%', markerfacecolor=rgba2, markersize=10),
                  Line2D([0], [0], marker='s', color=rgba3, label='95%', markerfacecolor=rgba3, markersize=10)]

# Add a legend to the plot
fig.legend(handles=legend_handles, loc='center')
# fig.legend(e1_plot, '68%')
plt.tight_layout()
fig.suptitle('phase shifts and LECs at different energies correlation')
plt.savefig('phase_shift_LEC_correlations_l_0_x.pdf')
plt.show()