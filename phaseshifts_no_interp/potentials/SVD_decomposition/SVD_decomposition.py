import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import matplotlib
import os
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.rcParams.update({'font.size': 14})
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

def path(order, S, L, Lprime, J, T, lamb, Nrows):
    file = "VNN_" + order + "_EM500new_SLLJT_%s%s%s%s%s_lambda_%s_Np_%s_np_nocut.dat" % (S, L, Lprime, J, T, lamb, Nrows)
    return file

S =      [1, 1, 1, 1]
L =      [0, 0, 2, 2]
Lprime = [0, 2, 0, 2]            #first index respectively 10010 3s1
J =      [1, 1, 1, 1]
T =      [0, 0, 0, 0]


S =      [0]
L =      [1]
Lprime = [1]           #1p1
J =      [1]
T =      [0]

S =      [0, 1, 0, 1, 1, 1]
L =      [0, 0, 1, 1, 1, 1]
Lprime = [0, 0, 1, 1, 1, 1]
J =      [0, 1, 1, 0, 1, 2]
T =      [1, 0, 0, 1, 1, 1]


SVD_range = 5
lamb = "2.00"

for o in range(5):
    #o iterates over all chiral orders
    for i in range(len(S)):
        #i iterates over the four partial wave channels
        p = path(orders[o], S[i], L[i], Lprime[i], J[i], T[i], lamb, 100)
        weights, nodes, potential = read_potential_from_file(p, 100)
        A, R, B = np.linalg.svd(potential)
        singular_values = np.empty((SVD_range))
        #SVD_sum = 0
        for l in range(SVD_range):
            #l iterates over the SVD orders
            SVD_part =  np.outer(A[:,l], B[l,:]) #* R[l]
            #SVD_sum+= SVD_part
            singular_values[l] = R[l]
            #plt.plot(nodes, np.diag(SVD_sum), label='SVD')
            #plt.plot(nodes, np.diag(potential), label='potential')
            #plt.legend()
            #plt.show()
            file_name = "SVD_" + "chiral_order_" + str(orders[o]) + "_lambda_" + lamb + "_SLLJT_" + str(S[i]) + str(L[i]) + str(Lprime[i]) + str(J[i]) + str(T[i]) + "_SVD_order_" + str(l+1)
            # file_name = "energies"
            f = open('/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/potentials/SVD_files/operators/' + file_name, 'w')
            for m in range(100):
                f.write(str(weights[m]) + " " + str(nodes[m]) + "\n")

            for a in range(100):
                for b in range(100):
                    f.write(str(nodes[a]) + " " + str(nodes[b]) + " " + str(SVD_part[a][b]) + "\n")

            f.close()

        file_name_singular_value = "SVD_" + "chiral_order_" + str(orders[o]) + "_lambda_" + lamb + "_SLLJT_" + str(S[i]) + str(L[i]) + str(Lprime[i]) + str(J[i]) + str(T[i]) + "_singular_values"

        k = open('/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/potentials/SVD_files/singular_values/' + file_name_singular_value, 'w')
        for n in range(SVD_range):
            k.write(str(singular_values[n]) + "\n")


        k.close()
