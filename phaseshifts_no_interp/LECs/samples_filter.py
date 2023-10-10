import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

dictionary_titles = {
    "00001": r'$^1$S$_0$',
    "10010": r'$^3$S$_1$',
    "01110": r'$^1$P$_1$',
    "11101": r'$^3$P$_0$',
    "11111": r'$^3$P$_1$',
    "11121": r'$^3$P$_2$'
}

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
energy_lin = np.linspace(energy[0], 200, 200)
SVD_rank = 4
def sample_filter(partial_wave, ratio):
    path_ = '/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/random_sampling/phaseshifts_SLLJT_%s_lambda_2.00_s5_new_1.dat' % (partial_wave)
    data_ = np.loadtxt(path_)

    LECs = data_[:, 0:5]
    ratios = data_[:,5]
    phaseshifts = data_[:,6:]
    phaseshifts_interp = np.zeros((len(phaseshifts[:, 0]), grid_size))
    for k in range(len(phaseshifts[:, 0])):
        ps_interpolate = sc.interpolate.interp1d(energy_, phaseshifts[k, :])
        phaseshifts_interp[k] = ps_interpolate(energy_lin)

    error_path = '/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/phaseshift_files/EKM_uncertainty/phaseshifts_uncertainties_SLLJT_%s_lambda_2.00_s5.dat' % (partial_wave)
    errors = np.loadtxt(error_path)

    N3LO_error = errors[:, 3]
    error_interpolate = sc.interpolate.interp1d(energy_, N3LO_error)
    error_interpolated = error_interpolate(energy_lin)

    reference_path = '/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/phaseshift_files/phaseshifts_SVD/phaseshifts_unchanged_SLLJT_%s_lambda_2.00_s5.dat' % (partial_wave)
    reference_data = np.loadtxt(reference_path)
    reference = reference_data[:, 3]
    reference_interpolate = sc.interpolate.interp1d(energy_, reference)
    reference_interpolated = reference_interpolate(energy_lin)
    ###remove when using NN generated phase shifts
    new_points = len(phaseshifts[:, 0])
    new_points = 15000

      # default 0.68

    check = 0
    phaseshift_set_ratio = np.zeros((new_points, grid_size))
    LEC_set_ratio = np.zeros((new_points, 5))
    indices = np.zeros((new_points, 1))
    ###replaces NN generated phase shift set with calculated one
    # phaseshift_set = phaseshifts_interp

    phaseshift_set = np.zeros((len(phaseshifts_interp[:, 0]), grid_size))
    # use this one for the data phase shifts

    #for k in range(len(phaseshifts_interp[:, 0])):
    #    ps_interpolate_lin = sc.interpolate.interp1d(energy, phaseshifts_interp[k, :])
    #    phaseshift_set[k] = ps_interpolate_lin(energy_lin)
    phaseshift_set = phaseshifts_interp

    LEC_set = LECs

    '''
    # use this one for the NN phaseshifts
    
    for k in range(len(phaseshift_set_[:, 0])):
        ps_interpolate_lin_NN = sc.interpolate.interp1d(energy, phaseshift_set_[k, :])
        phaseshift_set[k] = ps_interpolate_lin_NN(energy_lin)
    '''

    for m in range(new_points):
        count = 0
        for n in range(grid_size):

            if reference_interpolated[n] - np.abs(error_interpolated[n]) <= phaseshift_set[m, n] <= reference_interpolated[
                n] + np.abs(error_interpolated[n]):
                count += 1

        #print(count/grid_size)
        if count / grid_size > ratio:
            # phaseshift_set_ratio = np.delete(phaseshift_set, m, axis=0)
            # LEC_set_ratio = np.delete(LEC_set, m, axis=0)
            phaseshift_set_ratio[check] = phaseshift_set[m]
            LEC_set_ratio[check] = LEC_set[m]
            indices[check] = m
            check += 1

    phaseshift_set_ratio = np.delete(phaseshift_set_ratio, np.s_[check:new_points], axis=0)
    LEC_set_ratio = np.delete(LEC_set_ratio, np.s_[check:new_points], axis=0)
    indices = np.delete(indices, np.s_[check:new_points], axis=0)

    phaseshifts_filter = np.zeros((len(indices), grid_size))
    #for i, index in enumerate(indices):
    #    phaseshifts_filter[i] = phase
    ### write phaseshifts and LECs into a file here
    ### use indices to get original noninterpolated phaseshifts here
    file_name = "phaseshifts_SLLJT_%s_lambda_2.00_np_s%s_filter.dat" % (partial_wave, SVD_rank + 1)
    f = open('./samples_after_filter/' + file_name, 'w')
    #for k in range(len(indices)):
    print(np.shape(LEC_set_ratio))
    print(np.shape(phaseshift_set_ratio))
    results = np.column_stack((LEC_set_ratio, phaseshift_set_ratio))
    print(np.shape(results))
    for l in range(len(indices)):
        for m in range(len(results[0,:])):
            f.write(str(results[l, m]) + ' ')
        f.write('\n')
    f.close()
    for phaseshifts_ in phaseshift_set_ratio:
        plt.plot(energy_lin, phaseshifts_ / reference_interpolated, color='grey', linewidth=0.1, alpha=1)
    plt.fill_between(energy_lin, (reference_interpolated - error_interpolated) / reference_interpolated,
                     (reference_interpolated + error_interpolated) / reference_interpolated, color='orange', alpha=0.4)
    plt.plot(energy_lin, (reference_interpolated - error_interpolated) / reference_interpolated, linestyle='dashed',
             color='orange', alpha=0.6)
    plt.plot(energy_lin, (reference_interpolated + error_interpolated) / reference_interpolated, linestyle='dashed',
             color='orange', alpha=0.6)
    plt.ylim(0.8, 1.2)
    plt.xlim(0, 200)
    plt.title(dictionary_titles[partial_wave])
    plt.savefig('./plots_phaseshift_samples/' + partial_wave + '_68percent.pdf')
    plt.show()
    return phaseshift_set_ratio

phaseshift_set = sample_filter('00001', 0.68)









