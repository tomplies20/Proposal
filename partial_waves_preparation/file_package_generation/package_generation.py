import numpy as np
import os
import shutil

loc = '/Users/pleazy/PycharmProjects/Proposal/partial_waves_preparation/potentials_after_sampling_partial_waves/'
#loc = '/Users/pleazy/PycharmProjects/Proposal/partial_waves_preparation/samples_100_3s1/'
#final_loc = './packages/'
final_loc = '/Users/pleazy/Documents/Uni/Proposal/Pycharm/Proposal_project/'

files = os.listdir(loc)

#file.split('_')
#indicator[5] == minus

sign = ["plus", "minus"]

partial_waves = ['00001', '01110', '10010', '11101', '11111', '11121']



for i in range(len(files)):
    filename = files[i]
    src = loc + filename
    dst = final_loc + filename
    shutil.copy(src, dst)

'''
y = 1
for x1 in range(2):
    for x2 in range(2):
        for x3 in range(2):
            for x4 in range(2):
                for x5 in range(2):
                    for x6 in range(2):
                        #6 dateien mit _sign_ drin werden mit index y gespeichert
                        indices = [x1, x2, x3, x4, x5, x6]
                        print(y)
                        for i in range(6):
                            if partial_waves[i] == '00001':
                                #print(partial_waves[i])
                                print(sign[indices[i]])
                            #indicator = filename.split('_')
                            #if filename[61] == 'm':
                            filename = 'VNN_N3LO_Plies_SVD_s1_SLLJT_%s_lambda_2.00_Np_100_np_nocut_%s.dat' % (partial_waves[i], sign[indices[i]])
                            src = loc + "/" + filename
                            dst = final_loc + "/" + filename[0:14] + str(y) + filename[21:61] + '.dat'
                            #dst = final_loc + "/" +  str(y) + filename
                            #dst = final_loc + "/" + str(y) + filename
                            shutil.copy(src, dst)
                        y+=1
'''