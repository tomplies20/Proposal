import numpy as np
import os
from scipy.interpolate import interp2d


#### the file created by this code is a sum of all partial waves
### excluding s and p waves

potentials_path = '/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/potentials/files_lambda_2.00/N3LO'


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

temp = 0

for x in range(1):

    chiral_order = 'N3LO'  # change chiral order here!!
    root = "/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/potentials/files_lambda_2.00/" + chiral_order
    dirlist = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]
    for dir in dirlist:
        for filename in os.listdir(root + "/" + dir):
            indicator = filename.split('_')
            #print(dir)
            #print(indicator)
            #print(len(indicator))
            # print(filename)
            check = False
            #print(indicator[8])
            if indicator[8] == '100' and len(indicator) > 10:
                if indicator[9] == 'np' or indicator[9] == 'nn':
                    check = True
            if indicator[8] == '100' and len(indicator) > 11:
                if indicator[9] == 'pp':
                    check = True

            if check == True:
                pwave = indicator[4]
                if all ([pwave != '00001', pwave != '01110', pwave != '10010', pwave != '11101', pwave != '11111', pwave != '11121', ]):
                    print(pwave)
                #if pwave != 'madethisup':
                    # if pwave == '10010':
                    print('made it ')
                    weights, nodes, pot_ = read_potential_from_file('/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/potentials/files_lambda_2.00/' + chiral_order + '/' + dir + "/" + filename, 100)
                    begin = nodes[0]  # previously 0 and 6
                    end = nodes[len(nodes) - 1]
                    new_mesh_size = 100
                    step_width = (end - begin) / new_mesh_size
                    x_fine = np.linspace(begin, end, new_mesh_size)
                    y_fine = np.linspace(begin, end, new_mesh_size)

                    _nodes = x_fine
                    _weights = np.full(100, 6 / 100)
                    f_z = interp2d(nodes, nodes, pot_, kind='cubic')  ##########################big difference

                    pot_inter = f_z(x_fine, y_fine)
                    #temp += pot_inter

                    inds = filename.split('_')
                    print(inds)
                    inds[2] = 'Plies' #+ str(x)

                    new_filename = '_'.join(inds)
                    print(new_filename)

                    f = open('./potentials_not_s_p/' + new_filename, 'w')
                    for m in range(100):
                        f.write(str(_weights[m]) + " " + str(_nodes[m]) + "\n")

                    for i in range(100):
                        for j in range(100):
                            f.write(str(_nodes[i]) + " " + str(_nodes[j]) + " " + str(pot_inter[i][j]) + "\n")

                    f.close()

