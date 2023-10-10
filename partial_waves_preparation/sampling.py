import numpy as np
import os
import shutil
from numpy import loadtxt, sqrt, zeros, array, exp
from numpy import linalg
import random
### this script ist supposed to randomly sample sets of LECS obtained from
#### random phaseshift sampling for all partial waves (of interest) and all interactions (np, nn, pp)
#### and calculate the corresponding potential

partial_waves = ['00001', '10010', '01110', '11101', '11111', '11121']
interactions = ['np']
chiral_order = ['N3LO']
SVD_order = 4


def read_Vchiral(file, Nrows):

    mesh = np.genfromtxt(file, dtype=(float, float), skip_header=0, max_rows=Nrows)
    mesh_weights = mesh[:, 0]
    mesh_points = mesh[:, 1]



    Vread = np.genfromtxt(file, dtype=(float, float, float), skip_header=Nrows, max_rows=Nrows * Nrows)
    V = zeros([Nrows, Nrows], float)
    Vmat = zeros([Nrows, Nrows], float)
    Tkin = zeros([Nrows, Nrows], float)

    for i in range(Nrows):
        Tkin[i, i] = mesh_points[i] * mesh_points[i]
        for j in range(Nrows):
            V[i, j] = Vread[i * Nrows + j][2]
            Vmat[i, j] = 2.0 / np.pi * sqrt(mesh_weights[i]) * sqrt(mesh_weights[j]) * mesh_points[i] * mesh_points[j] * \
                         V[i, j]
    return [V, Vmat, Tkin, mesh_points, mesh_weights]



'''
loc = '/Users/pleazy/PycharmProjects/pythonProject/SVD/SVD_variation_potentials'
final_loc = '/Users/pleazy/PycharmProjects/pythonProject/kai_files/final_data'

files = os.listdir(loc)
'''




partial_waves = ['00001', '01110', '10010', '11101', '11111', '11121']

##determines how many samples you are going to pick from the ratio filtered LEC sets
## caution: initial amount of LEC sets is 15000 / 30000 but after ratio filtering
## can be significantly lower:
## chose number_samples lower than that amount
#number_samples = 100

#LECs = np.zeros((number_samples, 5*len(partial_waves)*len(interactions)))




SVD_rank = 4

grid_size = 200

loc_samples = '/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/LECs/samples_after_filter/'

def sampling_one_partial_wave(number_samples, partial_wave, particles):
    filename = "phaseshifts_SLLJT_%s_lambda_2.00_%s_s%s_filter.dat" % (partial_wave, particles, SVD_rank + 1)
    data_ = np.loadtxt(loc_samples + filename)
    LECs = data_[:, 0:5]
    phaseshifts = data_[:, 5:]
    final_LECs = np.zeros((number_samples, SVD_rank + 1))
    final_phaseshifts = np.zeros((number_samples, grid_size))
    for i in range(number_samples):
        potential_sum = 0
        random_index = random.randint(0, len(LECs[:, 0]))
        final_phaseshifts[i] = phaseshifts[random_index]
        final_LECs[i] = LECs[random_index]

        for o in range(SVD_order):

            operators_path = '/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/potentials/SVD_files/operators/' + "SVD_chiral_order_N3LO_lambda_2.00_SLLJT_%s_SVD_order_%s" % (partial_wave, o+1)

            [V00, Vmat_1, Tkin_1, mesh_points, mesh_weights] = read_Vchiral(operators_path , 100)
            V00_new = np.copy(V00) * LECs[random_index,o]
            potential_sum += V00_new
        potential_sum
        ###write out potential
        filename_final = 'VNN_N3LO_Plies%s_SLLJT_%s_lambda_2.00_Np_100_%s_nocut.dat' % (i, partial_wave, particles)
        g = open('./potentials_after_sampling_partial_waves/' + filename_final, 'w')
        for m in range(100):
            g.write(str(mesh_weights[m]) + " " + str(mesh_points[m]) + "\n")

        for a in range(100):
            for b in range(100):
                g.write(str(mesh_points[a]) + " " + str(mesh_points[b]) + " " + str(potential_sum[a][b]) + "\n")

        g.close()

    filename_potentials = "samples_SLLJT_%s_lambda_2.00_s%s.dat" % (partial_wave, SVD_rank + 1)
    f = open('./samples_after_sampling_partial_waves/' + filename_potentials, 'w')

    results = np.column_stack((final_LECs, final_phaseshifts))


    for l in range(number_samples):
        for m in range(len(results[0, :])):
            f.write(str(results[l, m]) + ' ')
        f.write('\n')
    f.close()

    return

sampling_one_partial_wave(50, '00001', 'np')











