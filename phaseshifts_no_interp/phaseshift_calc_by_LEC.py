#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 16:05:07 2022

@author: tom & yannick
"""

# import of necessary modules
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt, sqrt, zeros, array, exp
from numpy import linalg
import scipy as sc
from scipy.integrate import odeint
import random

# physical constants
hbarc = 197.326
M = (938.272 + 939.565) / 2.0  # averaged neutron/proton mass in MeV
units_factor = hbarc * hbarc / M
Nrows = 100  # 100: accurate, SRG takes some time! 50: less accurate, but faster

# arrays for uncoupled interaction channels
V = zeros([Nrows, Nrows], float)
Vmat = zeros([Nrows, Nrows], float)

# arrays for coupled interaction channels
Vcoupled = zeros([2 * Nrows, 2 * Nrows], float)
Vmatcoupled = zeros([2 * Nrows, 2 * Nrows], float)

# matrices for kinetic energy
Tkin = zeros([Nrows, Nrows], float)
Tkincoupled = zeros([2 * Nrows, 2 * Nrows], float)


# read in nuclear interaction matrix elements for given uncoupled partial wave channels




def read_Vchiral_og(title, S, L, Lprime, J, T, Nrows):
    file ="./potentials/SVD_decomposition/" +  "VNN_" + title + "_SLLJT_%s%s%s%s%s_lambda_2.00_Np_%s_np_nocut.dat" % (S, L, Lprime, J, T, Nrows) ##changed nn to np_nocut, #changed 50.00 to 2.00
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


# routines for computing phase shifts by solving the Lippmann Schwinger equation

# offset for subtracting the pole
eps = 1e-3


# computation of phase shifts in uncoupled channels
def counterterm(pmax, pole):
    return np.arctanh(pole / pmax) / pole


def compute_phase_shifts(K, i):
    return 180.0 / np.pi * np.arctan(-mesh_points[i] * K[i, i])


# note that ambiguity regarding the brach of arctan exists, make sure that solution is continuous as a function of energy
def compute_phase_shifts_coupled(K, i):
    epsilon = np.arctan(2 * K[i, i + Nrows] / (K[i, i] - K[i + Nrows, i + Nrows])) / 2.0
    r_epsilon = (K[i, i] - K[i + Nrows, i + Nrows]) / (np.cos(2 * epsilon))
    delta_a = - np.arctan(mesh_points[i] * (K[i, i] + K[i + Nrows, i + Nrows] + r_epsilon) / 2.0)
    delta_b = - np.arctan(mesh_points[i] * (K[i, i] + K[i + Nrows, i + Nrows] - r_epsilon) / 2.0)

    epsilonbar = np.arcsin(np.sin(2 * epsilon) * np.sin(delta_a - delta_b)) / 2.0
    delta_1 = 180.0 / np.pi * (delta_a + delta_b + np.arcsin(np.tan(2 * epsilonbar) / (np.tan(2 * epsilon)))) / 2.0
    delta_2 = 180.0 / np.pi * (delta_a + delta_b - np.arcsin(np.tan(2 * epsilonbar) / (np.tan(2 * epsilon)))) / 2.0

    epsilonbar *= -180.0 / np.pi
    return [delta_1, delta_2, epsilonbar]


def delta(i, j):
    if (i == j):
        return 1
    else:
        return 0


def Elab(p):
    return 2 * p ** 2 * hbarc ** 2 / M


def mom(E):
    return np.sqrt(M * E / 2 / hbarc ** 2)


def compute_K_matrix(V, Nrows, pmax):
    A = zeros([Nrows + 1, Nrows + 1], float)
    K = zeros([Nrows, Nrows], float)

    for x in range(Nrows):
        pole = mesh_points[x] + eps

        for i in range(Nrows):
            for j in range(Nrows):
                A[i, j] = delta(i, j) - 2.0 / np.pi * mesh_weights[j] * V[i, j] * mesh_points[j] ** 2 / (
                            pole ** 2 - mesh_points[j] ** 2)

        sum = 0.0
        for i in range(Nrows):
            sum += mesh_weights[i] / (pole ** 2 - mesh_points[i] ** 2)

        for i in range(Nrows):
            A[Nrows, i] = - 2.0 / np.pi * mesh_weights[i] * mesh_points[i] ** 2 * V[x, i] / (
                        pole ** 2 - mesh_points[i] ** 2)
            A[i, Nrows] = + 2.0 / np.pi * V[i, x] * pole ** 2 * (sum - counterterm(pmax, pole))

        A[Nrows, Nrows] = 1 + 2.0 / np.pi * V[x, x] * pole ** 2 * (sum - counterterm(pmax, pole))

        bvec = zeros([Nrows + 1], float)
        for i in range(Nrows):
            bvec[i] = V[i, x]
        bvec[Nrows] = V[x, x]

        xvec = np.linalg.solve(A, bvec)

        for i in range(Nrows):
            K[i, x] = xvec[i]

    return K


def compute_K_matrix_coupled(V00, V01, V10, V11, Nrows, pmax):
    A = zeros([2 * Nrows + 2, 2 * Nrows + 2], float)
    K = zeros([2 * Nrows, 2 * Nrows], float)

    for x in range(Nrows):
        pole = mesh_points[x] + eps

        for i in range(Nrows):
            for j in range(Nrows):
                A[i, j] = delta(i, j) - 2.0 / np.pi * mesh_weights[j] * V00[i, j] * mesh_points[j] ** 2 / (
                            pole ** 2 - mesh_points[j] ** 2)
                A[i, j + Nrows] = - 2.0 / np.pi * mesh_weights[j] * V01[i, j] * mesh_points[j] ** 2 / (
                            pole ** 2 - mesh_points[j] ** 2)
                A[i + Nrows, j] = - 2.0 / np.pi * mesh_weights[j] * V10[i, j] * mesh_points[j] ** 2 / (
                            pole ** 2 - mesh_points[j] ** 2)
                A[i + Nrows, j + Nrows] = delta(i, j) - 2.0 / np.pi * mesh_weights[j] * V11[i, j] * mesh_points[
                    j] ** 2 / (pole ** 2 - mesh_points[j] ** 2)

        sum = 0.0
        for i in range(Nrows):
            sum += mesh_weights[i] / (pole ** 2 - mesh_points[i] ** 2)

        for i in range(Nrows):
            A[2 * Nrows, i] = - 2.0 / np.pi * mesh_weights[i] * mesh_points[i] ** 2 * V00[x, i] / (
                        pole ** 2 - mesh_points[i] ** 2)
            A[i, 2 * Nrows] = + 2.0 / np.pi * V00[i, x] * pole ** 2 * (sum - counterterm(pmax, pole))

            A[2 * Nrows, i + Nrows] = - 2.0 / np.pi * mesh_weights[i] * mesh_points[i] ** 2 * V01[x, i] / (
                        pole ** 2 - mesh_points[i] ** 2)
            A[i, 2 * Nrows + 1] = + 2.0 / np.pi * V01[i, x] * pole ** 2 * (sum - counterterm(pmax, pole))

            A[2 * Nrows + 1, i] = - 2.0 / np.pi * mesh_weights[i] * mesh_points[i] ** 2 * V10[x, i] / (
                        pole ** 2 - mesh_points[i] ** 2)
            A[i + Nrows, 2 * Nrows] = + 2.0 / np.pi * V10[i, x] * pole ** 2 * (sum - counterterm(pmax, pole))

            A[2 * Nrows + 1, i + Nrows] = - 2.0 / np.pi * mesh_weights[i] * mesh_points[i] ** 2 * V11[x, i] / (
                        pole ** 2 - mesh_points[i] ** 2)
            A[i + Nrows, 2 * Nrows + 1] = + 2.0 / np.pi * V11[i, x] * pole ** 2 * (sum - counterterm(pmax, pole))

        A[2 * Nrows, 2 * Nrows] = 1 + 2.0 / np.pi * V00[x, x] * pole ** 2 * (sum - counterterm(pmax, pole))
        A[2 * Nrows, 2 * Nrows + 1] = + 2.0 / np.pi * V01[x, x] * pole ** 2 * (sum - counterterm(pmax, pole))
        A[2 * Nrows + 1, 2 * Nrows] = + 2.0 / np.pi * V10[x, x] * pole ** 2 * (sum - counterterm(pmax, pole))
        A[2 * Nrows + 1, 2 * Nrows + 1] = 1 + 2.0 / np.pi * V11[x, x] * pole ** 2 * (sum - counterterm(pmax, pole))

        bvec = zeros([2 * Nrows + 2, 2], float)
        for i in range(Nrows):
            bvec[i, 0] = V00[i, x]
            bvec[i, 1] = V01[i, x]
            bvec[i + Nrows, 0] = V10[i, x]
            bvec[i + Nrows, 1] = V11[i, x]

        bvec[2 * Nrows, 0] = V00[x, x]
        bvec[2 * Nrows, 1] = V01[x, x]
        bvec[2 * Nrows + 1, 0] = V10[x, x]
        bvec[2 * Nrows + 1, 1] = V11[x, x]

        xvec = np.linalg.solve(A, bvec)

        for i in range(Nrows):
            K[i, x] = xvec[i, 0]
            K[i, x + Nrows] = xvec[i, 1]
            K[i + Nrows, x] = xvec[i + Nrows, 0]
            K[i + Nrows, x + Nrows] = xvec[i + Nrows, 1]

    return K


def genenerate_wavename(S, L, J):
    dictio = {1: "P", 0: "S", 2: "D"}
    i = str(2 * S + 1)
    j = dictio[L]
    k = str(J)
    return "delta" + i + j + k + ".txt"


# read in particular uncoupled or coupled matrix elements
#title2 = "NLO_DeltaGO450"
title2 = "LO_EM500new"
S =      [0]
L =      [0]
Lprime = [0]            #first index respectively 10010 3s1
J =      [0]
T =      [1]
SVD_rank = 4 #+1 ##replace all (literal) percent arrays by np.zeros([SVD_rank+1])
#percent_plus_ = [0.01, 0, 0, 0, 0]
#percent_minus_ = [-0.015, 0, 0, 0, 0]


grid_size = 100

partial_wave = "$^1$D$_2$"
partial_wave = str(S[0]) + str(L[0]) + str(Lprime[0]) + str(J[0]) + str(T[0])

Nrows = 100
colors = ['r', 'navy', 'orange', 'teal', 'forestgreen']


begin = 0
end = 6
new_mesh_size = 100
step_width = (end- begin) / new_mesh_size
x_fine = np.linspace(begin + step_width/2, end - step_width/2, new_mesh_size)
y_fine = np.linspace(begin + step_width/2, end - step_width/2, new_mesh_size)

title3 = 'LO_fit'







pot_3s1 = 'LO_EM500new'


[V_LO, Vmat_1, Tkin_1, mesh_points, mesh_weights] = read_Vchiral_og('LO_EM500new', S[0], L[0], Lprime[0], J[0], T[0], Nrows)
#[V_NLO, Vmat_1, Tkin_1, mesh_points, mesh_weights] = read_Vchiral_og('NLO_EM500new', S[0], L[0], Lprime[0], J[0], T[0], Nrows)
#[V_N2LO, Vmat_1, Tkin_1, mesh_points, mesh_weights] = read_Vchiral_og('N2LO_EM500new', S[0], L[0], Lprime[0], J[0], T[0], Nrows)
#[V_N3LO, Vmat_1, Tkin_1, mesh_points, mesh_weights] = read_Vchiral_og('N3LO_EM500new', S[0], L[0], Lprime[0], J[0], T[0], Nrows)
#[V_N4LO, Vmat_1, Tkin_1, mesh_points, mesh_weights] = read_Vchiral_og('N4LO_EM500new', S[0], L[0], Lprime[0], J[0], T[0], Nrows)




#[V_fit, Vmat, Tkin, mesh_points, mesh_weights] = read_Vchiral(title3, S[0], L[0], Lprime[0], J[0], T[0], Nrows)


def phase_shift_correct(phase_shifts):
    p = np.copy(phase_shifts)
    found = False
    l = len(phase_shifts)
    for i in range(l - 2):
        if np.abs(phase_shifts[l - i - 2] - phase_shifts[l - i - 1]) > 50:  # previously < 1
            index = l - i - 1
            found = True
            # print(index)
            break
    if found == True:
        for m in range(index):
            phase_shifts[m] = np.copy(phase_shifts[m]) + 180
    return phase_shifts
    #return phase_shifts


'''
K_LO = compute_K_matrix(V_LO, Nrows, 6)
K_NLO = compute_K_matrix(V_NLO, Nrows, 6)
K_N2LO = compute_K_matrix(V_N2LO, Nrows, 6)
K_N3LO = compute_K_matrix(V_N3LO, Nrows, 6)
K_N4LO = compute_K_matrix(V_N4LO, Nrows, 6)
LO_phaseshifts = np.array([compute_phase_shifts(K_LO, i) for i in range(Nrows)])
NLO_phaseshifts = np.array([compute_phase_shifts(K_NLO, i) for i in range(Nrows)])
N2LO_phaseshifts = np.array([compute_phase_shifts(K_N2LO, i) for i in range(Nrows)])
N3LO_phaseshifts = np.array([compute_phase_shifts(K_N3LO, i) for i in range(Nrows)])
N4LO_phaseshifts = np.array([compute_phase_shifts(K_N4LO, i) for i in range(Nrows)])
'''
energy = Elab(mesh_points)


file_name_ps = "phaseshifts_unchanged_SLLJT_%s%s%s%s%s_lambda_2.00_s%s.dat" % (S[0], L[0], Lprime[0], J[0], T[0], SVD_rank + 1)
file_name_errors = "phaseshifts_uncertainties_SLLJT_%s%s%s%s%s_lambda_2.00_s%s.dat" % (S[0], L[0], Lprime[0], J[0], T[0], SVD_rank + 1)
ps = np.loadtxt('./phaseshift_files/phaseshifts_unchanged/' + file_name_ps)
LO_phaseshifts = ps[:,0]
NLO_phaseshifts = ps[:,1]
N2LO_phaseshifts = ps[:,2]
N3LO_phaseshifts = ps[:,3]
N4LO_phaseshifts = ps[:,4]

##
LO_phaseshifts = phase_shift_correct(LO_phaseshifts)
NLO_phaseshifts = phase_shift_correct(NLO_phaseshifts)
N2LO_phaseshifts = phase_shift_correct(N2LO_phaseshifts)
N3LO_phaseshifts = phase_shift_correct(N3LO_phaseshifts)
N4LO_phaseshifts = phase_shift_correct(N4LO_phaseshifts)


errors = np.loadtxt('./phaseshift_files/EKM_uncertainty/' + file_name_errors)
error_LO_x = errors[:,0]
error_NLO_x = errors[:,1]
error_N2LO_x = errors[:,2]
error_N3LO_x = errors[:,3]
error_N4LO_x = errors[:,4]


phaseshifts_order = [LO_phaseshifts, NLO_phaseshifts, N2LO_phaseshifts, N3LO_phaseshifts, N4LO_phaseshifts]

errors_order = [error_LO_x, error_NLO_x, error_N2LO_x, error_N3LO_x, error_N3LO_x]

energy = Elab(mesh_points)


orders = ["LO", "NLO", "N2LO", "N3LO", "N4LO"]

def SVD(chiral_order, SVD_order, lamb,  percent ):
        potential_sum = 0
        sv00 = './potentials/SVD_files/singular_values/' + "SVD_chiral_order_%s_lambda_%s_SLLJT_%s%s%s%s%s_singular_values" % (
            orders[chiral_order], lamb ,S[0], L[0], Lprime[0], J[0], T[0])
        singular_values = np.loadtxt(sv00)
        for o in range(SVD_order + 1):
            sv = singular_values[o]
            #if o in SVD_variation_index:
            sv = sv + sv * percent[o]
            V00_new = 0
            file00 = './potentials/SVD_files/operators/' + "SVD_chiral_order_%s_lambda_%s_SLLJT_%s%s%s%s%s_SVD_order_%s" % (orders[chiral_order], lamb, S[0], L[0], Lprime[0], J[0], T[0], o+1)

            [V00, Vmat_1, Tkin_1, mesh_points, mesh_weights] = read_Vchiral(file00 , Nrows)
            V00_new = np.copy(V00) * sv
            potential_sum += V00_new

        K = compute_K_matrix(potential_sum , Nrows, 6)
        phase_shifts = np.array([compute_phase_shifts(K, i) for i in range(Nrows)])
        phase_shifts = phase_shift_correct(phase_shifts)
        return phase_shifts


def SVD_potential(chiral_order, SVD_order, lamb, percent):
    potential_sum = 0
    sv00 = './potentials/SVD_files/singular_values/' + "SVD_chiral_order_%s_lambda_%s_SLLJT_%s%s%s%s%s_singular_values" % (
        orders[chiral_order], lamb, S[0], L[0], Lprime[0], J[0], T[0])
    singular_values = np.loadtxt(sv00)
    for o in range(SVD_order + 1):
        sv = singular_values[o]
        # if o in SVD_variation_index:
        sv = sv + sv * percent[o]
        V00_new = 0
        file00 = './potentials/SVD_files/operators/' + "SVD_chiral_order_%s_lambda_%s_SLLJT_%s%s%s%s%s_SVD_order_%s" % (
        orders[chiral_order], lamb, S[0], L[0], Lprime[0], J[0], T[0], o + 1)

        [V00, Vmat_1, Tkin_1, mesh_points, mesh_weights] = read_Vchiral(file00, Nrows)
        V00_new = np.copy(V00) * sv
        potential_sum += V00_new

    K = compute_K_matrix(potential_sum, Nrows, 6)
    phase_shifts = np.array([compute_phase_shifts(K, i) for i in range(Nrows)])
    phase_shifts = phase_shift_correct(phase_shifts)
    return potential_sum





#'''



#NLO_fit_phaseshifts = np.loadtxt("phaseshifts_NLO_3S1_fit")


M = (938.272 + 939.565) / 2.0

pion_momentum = 139.57039 / hbarc * np.ones([len(energy)])



reference = N3LO_phaseshifts



alp = 0.2








def variation_plus(chiral_order, SVD_order, lamb, percent):

    SVD_plus_x = SVD(chiral_order, SVD_order, lamb, percent)
    SVD_x = SVD(chiral_order, SVD_order, lamb, [0, 0, 0, 0, 0])
    reference_x = phaseshifts_order[chiral_order]
    xx = np.linspace(energy[0], 200 + energy[0], grid_size ) #previously len(energy) + 1
    energy_inter = xx

    f_plus = sc.interpolate.interp1d(energy, SVD_plus_x)
    plus_inter = f_plus(xx)
    SVD_plus = f_plus(xx)



    f_ = sc.interpolate.interp1d(energy, SVD_x)
    _inter = f_(xx)
    SVD_ = f_(xx)



    f_ref = sc.interpolate.interp1d(energy, reference_x)
    ref_inter = f_ref(xx)
    reference = f_ref(xx)




    f_err = sc.interpolate.interp1d(energy, errors_order[chiral_order])
    err_inter = f_err(xx)
    error_ = f_err(xx)

    ratios = np.zeros([3])
    max_energies = np.array([20, 50, 200])

    count = 0
    for max_energy in max_energies:
        l = 0
        sum_ = 0
        index_ = 0
        while energy_inter[index_] < max_energy:

            if SVD_plus[index_] - reference[index_] > 0:
                if SVD_plus[index_] < reference[index_] + np.abs(error_[index_]):
                    sum_ += 1
                index_ += 1
            else:
                if SVD_plus[index_] > reference[index_] - np.abs(error_[index_]):
                    sum_ += 1
                index_ += 1



        ratio = sum_ / (index_)
        ratios[l] = ratio
        l+=1
    print('energy limit: ' + str(max_energy) + 'MeV' + ' ratio: ' + str(ratio))



    return SVD_plus, reference, error_, ratio





def variation2(chiral_order, SVD_order, lamb, percent_plus, percent_minus, sv_index, ps_plus, ps_minus):

    SVD_plus_x = SVD(chiral_order, SVD_order, lamb, percent_plus)
    SVD_x = SVD(chiral_order, SVD_order, lamb, [0, 0, 0, 0, 0])
    SVD_minus_x = SVD(chiral_order, SVD_order, lamb,  -1 * percent_minus)
    reference_x = phaseshifts_order[chiral_order]
    xx = np.linspace(energy[0], 200 + energy[0], grid_size) #previously len(energy) + 1
    energy_inter = xx


    f_plus = sc.interpolate.interp1d(energy, SVD_plus_x)

    SVD_plus = f_plus(xx)

    f_ = sc.interpolate.interp1d(energy, SVD_x)
    _inter = f_(xx)
    SVD_ = f_(xx)

    f_minus = sc.interpolate.interp1d(energy, SVD_minus_x)

    SVD_minus = f_minus(xx)

    f_ref = sc.interpolate.interp1d(energy, reference_x)

    reference = f_ref(xx)

    f_err = sc.interpolate.interp1d(energy, errors_order[chiral_order])

    error_ = f_err(xx)

    plt.figure(figsize=(10, 10))

    plt.fill_between(energy_inter, (reference + error_)/reference, (reference - error_)/reference, color='purple', alpha=alp,
                     label=orders[chiral_order]+' error')
    #plt.plot(energy_inter, SVD_, label='s' + str(sv_index + 1) + ' + ' + "0%")
    color_arr = ['blue', 'purple', 'red', 'orange', 'yellow']
    for n in range(len(ps_plus)):

        plt.plot(energy_inter, ps_plus[n,:]/reference, label='s' + str(n + 1) , color=color_arr[n], linewidth=0.5)  # 1 = NLO

        plt.plot(energy_inter, ps_minus[n,:]/reference, label='s' + str(n + 1), color=color_arr[n], linewidth=0.5, linestyle='dashed')
    plt.ylim(0.9, 1.1)
    plt.plot(energy_inter, reference/reference, label=orders[chiral_order], color='black', linestyle='dashed', linewidth=1)

    #plt.vlines(max_energy, -25, 5, color='grey')

    plt.xlabel('E (MeV)')
    plt.ylabel(r'$\delta$ (°) ')

    plt.xlim([0, 200])
    # plt.xlim([0, 20])
    #plt.ylim([-25,5])
    # plt.ylim([-3, 0])
    plt.legend()
    #plt.ylim(-25, -10)
    plt.title(orders[chiral_order] + " SVD phaseshifts " + partial_wave + " singular value s" + str(sv_index + 1))
    plt.savefig("./phaseshift_plots_all_singular_values/" + orders[chiral_order] + "_" + partial_wave +"_SVD_phaseshifts_sv_" +'phaseshifts'+ ".pdf")
    plt.show()




step_size = 0.001












#perc_steps = np.array([0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1])

perc_steps = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 1, -0.05, -0.1, -0.25, -0.5, -0.75, -1])

phaseshifts_ = np.zeros([SVD_rank + 1, 1, grid_size])
#perc_steps = np.array([0.5])
#percent_plus = np.zeros([5])
#1S0

percent_minus = [0, 2.05, -0.04, 0, -0.3]


def random_sampling(number_points, chiral_order):
    sv00 = './potentials/SVD_files/singular_values/' + "SVD_chiral_order_%s_lambda_%s_SLLJT_%s%s%s%s%s_singular_values" % (
        orders[chiral_order], '2.00', S[0], L[0], Lprime[0], J[0], T[0])
    singular_values = np.loadtxt(sv00)[:5]
    file_name = "phaseshifts_SLLJT_%s%s%s%s%s_lambda_2.00_s%s_2.dat" % (S[0], L[0], Lprime[0], J[0], T[0], SVD_rank + 1)
    f = open('./random_sampling/' + file_name, 'w')
    for i in range(number_points):
        percent= [0, random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
        phaseshifts, x_, y_, ratio = variation_plus(3, SVD_rank, "2.00", percent)
        ratio_array = np.array([ratio])
        print(percent)
        print(ratio)
        new_singular_values = singular_values + singular_values * percent
        results = np.concatenate((ratio_array, phaseshifts))
        results = np.concatenate((new_singular_values, results))
        for m in results:
            f.write(str(m) + ' ')
        f.write('\n')

percent = [0, 0, 0.5, 0, 0]
phaseshift_by_percent = variation_plus(3, SVD_rank, "2.00", percent)[0]
#random_sampling(10000, 3)
print(phaseshift_by_percent)