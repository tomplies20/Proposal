import numpy as np
from scipy.interpolate import interp2d
import os

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

'''
file_name = 'VNN_%s_EM500new_SLLJT_%s%s%s%s%s_lambda_2.00_Np_100_np_nocut.dat' % (chiral_order, S, L, Lprime, J, T)
#file_name = '3d1_NLO_fit_pot.dat'

weights, nodes , pot_ = read_potential_from_file('unterinterpolated_potentials/' + file_name, 100)

begin = nodes[0] #previously 0 and 6
end = nodes[len(nodes)-1]
new_mesh_size = 100
step_width = (end- begin) / new_mesh_size
x_fine = np.linspace(begin, end, new_mesh_size)
y_fine = np.linspace(begin, end, new_mesh_size)


_nodes = x_fine
_weights = np.full(100, 6 / 100)

f_z = interp2d(nodes, nodes, pot_, kind='cubic')  ##########################big difference

pot_inter = f_z(x_fine, y_fine)
'''




'''
f = open('/Users/pleazy/PycharmProjects/pythonProject/phaseshifts/interpolated_potentials_new_grid/' + file_name, 'w')
for m in range(100):
    f.write(str(_weights[m]) + " " + str(_nodes[m]) + "\n")

for i in range(100):
    for j in range(100):
        f.write(str(_nodes[i]) + " " + str(_nodes[j]) + " " + str(pot_inter[i][j]) + "\n")

f.close()
'''

#range eigentlich 65!!!!
for x in range(50):
    #for dir in os.listdir("/Users/pleazy/PycharmProjects/pythonProject/kai_files/data"):
    #    for filename in dir:
    #        print(dir)
    #        indicator = filename.split("_")
    #        print(indicator)
    chiral_order = 'N3LO' #change chiral order here!!
    root = "/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/potentials/files_lambda_2.00/" + chiral_order
    dirlist = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]
    for dir in dirlist:
        for filename in os.listdir(root + "/" + dir):
            indicator = filename.split('_')
            print(dir)
            print(indicator)
            print(len(indicator))
            #print(filename)
            check = False
            print(indicator[8])
            if indicator[8] == '100' and len(indicator) > 10 :
                if indicator[9] == 'np' or indicator[9] == 'nn':
                    check = True
            if indicator[8] == '100' and len(indicator) > 11:
                if indicator[9] == 'pp':
                    check = True

            if check == True:
                pwave =  indicator[4]
                #if all ([pwave != '00001', pwave != '01110', pwave != '10010', pwave != '11101', pwave != '11111', pwave != '11121', ]):
                if pwave != 'madethisup':
                #if pwave == '10010':
                    print('made it ')
                    weights, nodes , pot_ = read_potential_from_file('/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/potentials/files_lambda_2.00/' + chiral_order +'/' +dir + "/" + filename, 100)
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
                    inds = filename.split('_')
                    print(inds)
                    inds[2] = 'Plies' + str(x)
                    #inds[1] = 'N3LO' #only temporary
                    new_filename = '_'.join(inds)
                    print(new_filename)
                    #new_filename = filename[0:9] + 'Plies' + str(x) + filename[17:]

                    f = open('./packages/' + new_filename, 'w')
                    for m in range(100):
                        f.write(str(_weights[m]) + " " + str(_nodes[m]) + "\n")
            
                    for i in range(100):
                        for j in range(100):
                            f.write(str(_nodes[i]) + " " + str(_nodes[j]) + " " + str(pot_inter[i][j]) + "\n")
                            #f.write(str(_nodes[i]) + " " + str(_nodes[j]) + " " + str(0) + "\n") #for zero files
            
                    f.close()
