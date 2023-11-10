import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import matplotlib
import os
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.rcParams.update({'font.size': 14})



### careful:
### other than SVD_decomposition this script works with potentials that
### have NOT been interpolated yet ! ! !


def read_potential_from_file(path_to_file, number_of_mesh_points):
    # Weights and nodes for the quadrature
    weights = np.zeros(number_of_mesh_points)
    nodes = np.zeros(number_of_mesh_points)

    # Potential matrix elements
    potential = np.zeros((number_of_mesh_points, number_of_mesh_points))

    with open(path_to_file) as f:
        for i in range(number_of_mesh_points):
            line = f.readline()
            line_terms = line.split()
            w = float(line_terms[0])
            p = float(line_terms[1])
            weights[i] = w
            nodes[i] = p

        for i in range(number_of_mesh_points):
            for j in range(number_of_mesh_points):
                line = f.readline()
                line_terms = line.split()
                potential[i][j] = float(line_terms[2])

    return weights, nodes, potential

orders = ["LO", "NLO", "N2LO", "N3LO", "N4LO"]

def path(order ,  S, L, Lprime, J, T, lamb, Nrows, interaction):
    file = "VNN_%s_EM500new_SLLJT_%s%s%s%s%s_lambda_%s_Np_%s_%s_nocut.dat" % (order, S, L, Lprime, J, T, lamb, Nrows, interaction)
    return file

S =      [1, 1, 1, 1]
L =      [0, 0, 2, 2]
Lprime = [0, 2, 0, 2]            #first index respectively 10010 3s1
J =      [1, 1, 1, 1]
T =      [0, 0, 0, 0]


S =      [1]
L =      [1]
Lprime = [1]           #1p1
J =      [0]
T =      [1]

S =      [0,  1, 1, 1]
L =      [0,  1, 1, 1]
Lprime = [0,  1, 1, 1]
J =      [0,  0, 1, 2]
T =      [1,  1, 1, 1]





SVD_range =5
lamb = "2.00"

interaction = 'nn'
interaction = 'pp_Vc'

for o in range(3, 4):
    #o iterates over all chiral orders
    for i in range(len(S)):
        #i iterates over the four partial wave channels
        p = path(orders[o], S[i], L[i], Lprime[i], J[i], T[i], lamb, 100, interaction)
        weights, nodes, potential = read_potential_from_file(p, 100)
        begin = nodes[0]  # previously 0 and 6
        end = nodes[len(nodes) - 1]
        new_mesh_size = 100
        step_width = (end - begin) / new_mesh_size
        x_fine = np.linspace(begin, end, new_mesh_size)
        y_fine = np.linspace(begin, end, new_mesh_size)

        _nodes = x_fine
        _weights = np.full(100, 6 / 100)
        f_z = interp2d(nodes, nodes, potential, kind='cubic')  ##########################big difference

        pot_inter = f_z(x_fine, y_fine)
        A, R, B = np.linalg.svd(pot_inter)
        singular_values = np.empty((SVD_range))

        for l in range(SVD_range):
            #l iterates over the SVD orders
            SVD_part =  np.outer(A[:,l], B[l,:]) #* R[l]

            singular_values[l] = R[l]

            file_name = f'VNN_N3LO_s{l+1}Plies_SLLJT_{S[i]}{L[i]}{Lprime[i]}{J[i]}{T[i]}_lambda_2.00_Np_100_{interaction}_nocut.dat'
            #file_name = "SVD_" + "chiral_order_" + str(orders[o]) + "_lambda_" + lamb + "_SLLJT_" + str(S[i]) + str(L[i]) + str(Lprime[i]) + str(J[i]) + str(T[i]) + "_SVD_order_" + str(l+1)
            # file_name = "energies"
            f = open('./operators_pp_nn/' + file_name, 'w')
            for m in range(100):
                f.write(str(weights[m]) + " " + str(nodes[m]) + "\n")

            for a in range(100):
                for b in range(100):
                    f.write(str(nodes[a]) + " " + str(nodes[b]) + " " + str(SVD_part[a][b]) + "\n")

            f.close()

        #file_name_singular_value = "SVD_" + "chiral_order_" + str(orders[o]) + "_lambda_" + lamb + "_SLLJT_" + str(S[i]) + str(L[i]) + str(Lprime[i]) + str(J[i]) + str(T[i]) + "_singular_values"
        file_name_singular_values = f'VNN_N3LO_sv{l + 1}Plies_SLLJT_{S[i]}{L[i]}{Lprime[i]}{J[i]}{T[i]}_lambda_2.00_Np_100_{interaction}_nocut.dat'
        k = open('./singular_values_pp_nn/' + 'sv_' + file_name_singular_values , 'w')

        for n in range(SVD_range):
            k.write(str(singular_values[n]) + "\n")


        k.close()
