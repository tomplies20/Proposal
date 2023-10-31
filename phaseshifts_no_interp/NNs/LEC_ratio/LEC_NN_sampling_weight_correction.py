import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import torch


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import random
from scipy.integrate import quad

from phaseshift_calculator_LECs import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # choose the hardware accelerator



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
#energy_lin = np.linspace(energy_[0], 200, 200)
print(len(energy))




partial_wave = '10010'


data_ = np.loadtxt(f'/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/random_sampling/phaseshifts_SLLJT_{partial_wave}_lambda_2.00_s5_new_2.dat')
#data_ = np.loadtxt('/Users/pleazy/Desktop/phaseshifts_SLLJT_01110_lambda_2.00_s5.dat')

LECs = data_[:, 0:5]
ratios = data_[:,5]
phaseshifts = data_[:,6:]
phaseshifts_interp = np.zeros((len(phaseshifts[:, 0]), grid_size))
for k in range(len(phaseshifts[:, 0])):
    ps_interpolate = sc.interpolate.interp1d(energy_, phaseshifts[k, :])
    phaseshifts_interp[k] = ps_interpolate(energy)
#plt.plot(energy_, phaseshifts[100])
print('done')




labels = np.zeros((50000))

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # Data samples, e.g., images
        self.labels = labels  # Labels for each sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #sample = {
        #    'data': self.data[idx],  # A data sample (e.g., an image)
        #    'label': self.labels[idx]  # Corresponding label for the sample
        #}
        #return sample
        return self.data[idx,:], torch.tensor([labels[idx]])


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dims[i - 1], hidden_dims[i]) for i in range(1, len(hidden_dims))
        ])
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        x = nn.functional.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = nn.functional.relu(layer(x))

        #x = nn.functional.sigmoid(self.output_layer(x)) # prev x = self.output_layer(x)
        x = self.output_layer(x)
        return x



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dims[i - 1], hidden_dims[i]) for i in range(1, len(hidden_dims))
        ])
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        x = nn.functional.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = nn.functional.relu(layer(x))

        #x = nn.functional.sigmoid(self.output_layer(x)) # prev x = self.output_layer(x)
        x = self.output_layer(x)
        return x

input_dim = 6
output_dim = 1
hidden_dims = [12, 128, 128, 64, 256, 64, 256, 64, 32]

model = MLP(input_dim, output_dim, hidden_dims).to(device)
model.load_state_dict(torch.load(f'/Users/pleazy/Documents/Uni/Proposal/JupyterNotebook/model_{partial_wave}.pth'))
model.eval()










new_points = 240

ratio_NN = 0.8
ratio = 0.68

#lower limits, max weight is 0.5
#0.35 exhausts the lower energy part well, higher part is not fully covered

weight_limit_NN = 0
weight_limit = 0

phaseshift_set_ = np.zeros((new_points, grid_size))
LEC_set = np.zeros((new_points, 5))

old_data = np.loadtxt('/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/NNs/LEC_ratio/LEC_files/phaseshifts_SLLJT_10010_lambda_2.00_s5_new_2.dat')
old_weights = old_data[:, 205]
old_LECs = old_data[:, 0:5]

for i in range(new_points):
    sv_path = f'/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/potentials/SVD_files/singular_values/SVD_chiral_order_N3LO_lambda_2.00_SLLJT_{partial_wave}_singular_values'
    svs = np.loadtxt(sv_path)[0:5]
    #percentages = np.array([random.uniform(-0.0, 0.0), random.uniform(-0.0, 0.0), random.uniform(-0.25, 0.25), random.uniform(-1, 1), random.uniform(-1, 1)]) #11111 parameter space
    percentages = np.array([random.uniform(-0.02, 0.02), random.uniform(-0.2, 0.2), random.uniform(-0.25, 0.25), random.uniform(-1, 1), random.uniform(-1, 1)])
    #percentages = [0, 0, 0, 0, 0] #best possible total weight is 0.5
    random_LECs = old_LECs[i]
    dataset_ = np.zeros((grid_size, 6))
    for h in range(grid_size):
        dataset_[h] = [random_LECs[0], random_LECs[1], random_LECs[2], random_LECs[3], random_LECs[4], energy[h]]

    dataset_ = dataset_.astype(np.float32)

    len_train_ = 1

    testset_ = CustomDataset(dataset_, energy)
    #print(testset_.__len__())

    batch_size_ = grid_size

    testloader_= DataLoader(testset_, batch_size=batch_size_)

    with torch.no_grad():
        for data__ in testloader_:
            inputs_, targets_ = data__
            #print(data__)
            #print(inputs_)
            outputs_ = model(inputs_.to(device))
            #print(outputs_)
            break
    outputs = outputs_.numpy()
    phaseshift_set_[i] = outputs.ravel()
    LEC_set[i] = random_LECs




def gaussian(x, mu, sigma):
    return 1/(np.sqrt( 2 * np.pi) * sigma) * np.exp( -1/2 * (( x - mu )/sigma)**2)



errors = np.loadtxt(
    f'/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/phaseshift_files/EKM_uncertainty/phaseshifts_uncertainties_SLLJT_{partial_wave}_lambda_2.00_s5.dat')

N3LO_error = errors[:, 3]
error_interpolate = sc.interpolate.interp1d(energy_, N3LO_error)
error_interpolated = error_interpolate(energy_lin)

reference_data = np.loadtxt(
    f'/Users/pleazy/PycharmProjects/Proposal/phaseshifts_no_interp/phaseshift_files/phaseshifts_SVD/phaseshifts_unchanged_SLLJT_{partial_wave}_lambda_2.00_s5.dat')
reference = reference_data[:, 3]
reference_interpolate = sc.interpolate.interp1d(energy_, reference)
reference_interpolated = reference_interpolate(energy_lin)




def covariance(x, xprime, c,L ):
    return c**2 * np.exp(- 1/2 * (x - xprime).T * 1/L * (x - xprime))
covariance()

def ratio_test(weight_limit, phaseshift_set_):
      # default 0.68

    check = 0
    phaseshift_set_ratio = np.zeros((new_points, grid_size))
    LEC_set_ratio = np.zeros((new_points, 5))

    ##weights low energy phase shifts higher than high energy ones
    lin_space = np.linspace(1, 0, grid_size) #change back to (1,0, grid_size)

    # use this one for the NN phaseshifts
    phaseshift_set = np.zeros((new_points, grid_size))
    for k in range(len(phaseshift_set_[:, 0])):
        ps_interpolate_lin_NN = sc.interpolate.interp1d(energy, phaseshift_set_[k, :])
        phaseshift_set[k] = ps_interpolate_lin_NN(energy_lin)
    weights = np.zeros((new_points))
    for m in range(new_points):

        weight = 0
        count = 0
        for n in range(18, 19):
            #gauss = gaussian(phaseshift_set[m, n], reference_interpolated[n], np.abs(error_interpolated[n]))
            mu = reference_interpolated[n]
            sigma = np.abs(error_interpolated[n])
            distance = np.abs(phaseshift_set[m, n] - reference_interpolated[n])
            result, error = quad(gaussian, reference_interpolated[n] + distance, reference_interpolated[n] + distance+10*sigma, args=(mu, sigma))
            gauss = result
            weight += gauss * lin_space[n]
            #print(gauss)

            if reference_interpolated[n] - np.abs(error_interpolated[n]) <= phaseshift_set[m, n] <= reference_interpolated[
                n] + np.abs(error_interpolated[n]):
                count+=1
        print('ratio: ' + str(count/grid_size))


        print('avg weight: ' + str(weight/grid_size))
        if weight/grid_size >= weight_limit: #0.1 for (1, 0) additional weight factor
            phaseshift_set_ratio[check] = phaseshift_set[m]
            LEC_set_ratio[check] = LEC_set[m]
            weights[check] = weight / grid_size
            check += 1

    '''
    for m in range(new_points):
        count = 0
        count_40 = 0
        for n in range(grid_size):
            ####this is correct dont rethink it a third time
            if reference_interpolated[n] - np.abs(error_interpolated[n]) <= phaseshift_set[m, n] <= reference_interpolated[
                n] + np.abs(error_interpolated[n]):
                if n <40:
                    count_40+=1
                count += 1

        print(count_40/40)
        if count / grid_size  > ratio and count_40 / 40 > ratio:

            phaseshift_set_ratio[check] = phaseshift_set[m]
            LEC_set_ratio[check] = LEC_set[m]
            check += 1
    '''
    print('number of samples: ' + str(check))
    phaseshift_set_ratio = np.delete(phaseshift_set_ratio, np.s_[check:new_points], axis=0)
    weights = np.delete(weights, np.s_[check:new_points], axis=0)
    #print(np.shape(phaseshift_set_ratio))
    print(np.shape(weights))
    LEC_set_ratio = np.delete(LEC_set_ratio, np.s_[check:new_points], axis=0)
    return LEC_set_ratio, phaseshift_set_ratio, weights #LECs and linearly interpolated phase shifts that passed the ratio test



LEC_set_ratio_NN, phaseshifts_NN, weights_NN = ratio_test(weight_limit_NN, phaseshift_set_)
num_samples = len(LEC_set_ratio_NN[:,0])

for phaseshifts_ in phaseshifts_NN:
    plt.plot(energy_lin, phaseshifts_/reference_interpolated, color='grey', linewidth=0.1, alpha=1)
plt.fill_between(energy_lin, (reference_interpolated - error_interpolated)/reference_interpolated, (reference_interpolated + error_interpolated)/reference_interpolated, color='orange', alpha=0.4)
plt.plot(energy_lin, (reference_interpolated - error_interpolated)/reference_interpolated, linestyle='dashed', color='orange', alpha = 0.6)
plt.plot(energy_lin, (reference_interpolated + error_interpolated)/reference_interpolated, linestyle='dashed', color='orange', alpha = 0.6)
plt.ylim(0.8, 1.2)
plt.xlim(0, 200)
plt.show()

old_grid_size = 100
phaseshift_set_final = np.zeros((num_samples, old_grid_size))



for q in range(num_samples):
    phaseshift_set_final[q] = SVD(f'{partial_wave}', 3, 4, '2.00',  LEC_set_ratio_NN[q])
    print(q)
phaseshift_set_final_non_uniform_interpolation = np.zeros((num_samples, grid_size))

for j in range(num_samples):
    ps_interpolate_non_uniform = sc.interpolate.interp1d(energy_, phaseshift_set_final[j, :])
    phaseshift_set_final_non_uniform_interpolation[j] = ps_interpolate_non_uniform(energy)



LEC_set_final, phaseshift_set_final_interpolated, weights_ = ratio_test(weight_limit, phaseshift_set_final_non_uniform_interpolation)

file_name = f"new_weights_2.dat"
f = open('./LEC_files/' + file_name, 'w')

for i in range(len(weights_)):
    weights = np.atleast_1d(weights_)

    #results = np.column_stack((np.array(weight), phaseshift_set_final_interpolated[i,:]))
    #results = np.concatenate((LEC_set_final[i], phaseshift_set_final_interpolated[i,:]))
    #results = np.concatenate((results, [weights[i]]))
    results = weights[i]

    f.write(str(results) + ' ')
    f.write('\n')

for phaseshifts_ in phaseshift_set_final_interpolated:
    plt.plot(energy_lin, phaseshifts_/reference_interpolated, color='grey', linewidth=0.1, alpha=1)
plt.fill_between(energy_lin, (reference_interpolated - error_interpolated)/reference_interpolated, (reference_interpolated + error_interpolated)/reference_interpolated, color='orange', alpha=0.4)
plt.plot(energy_lin, (reference_interpolated - error_interpolated)/reference_interpolated, linestyle='dashed', color='orange', alpha = 0.6)
plt.plot(energy_lin, (reference_interpolated + error_interpolated)/reference_interpolated, linestyle='dashed', color='orange', alpha = 0.6)
plt.ylim(0.8, 1.2)
plt.xlim(0, 200)
#plt.savefig('phaseshift_relative_error_11111_68_percent.pdf')

plt.show()




fig, axes = plt.subplots(5, 5, figsize=(12, 12))#, sharex='col', sharey='row')

#weights = data[:, 205]
weights = weights_
for i in range(5):
    for j in range(5):
        # Only plot if we are in the left half (including the diagonal)
        if j <= i:



            # Plot the data on the current subplot
            axes[i, j].scatter(LECs[:,i], LECs[:,j], s=0.5,  alpha =(weights/max(weights))**6)#, s=(weights/max(weights))**4 )
            #print((weights/max(weights))**10)
            #axes[i, j].set_title(f'Plot {i+1}-{j+1}')
        else:
            axes[i, j].axis('off')
plt.savefig('LEC_correlations_10010_68_weights.pdf')
plt.show()


