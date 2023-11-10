import os
import numpy as np
import pandas as pd
# Directory containing the folders
directory_path = '/data_share4/ox_tom/all_operators'  # Replace with the actual directory path
directory_path = '/data_share4/ox_tom/one_operator'
higher_partial_waves_path = '/data_share4/ox_tom/transformed_not_s_p/hw_16.00'

singular_values_path = '/data_share4/ox_tom/singular_values_post'
output_path = '/data_share4/ox_tom/transformed_s_p'
# Function to count 'hw_16' files in a folder
def count_hw_16_files(folder_path):
    # Check if 'hw_16' file exists in the folder
    hw_16_file_path = os.path.join(folder_path, 'hw_16.00')
    return os.path.isdir(hw_16_file_path)

# Initialize a counter
count = 0

# List all items (folders and files) in the directory
dirs = os.listdir(directory_path)

final_array = 0
final_last_line = 0

# Iterate through items in the directory #10010_np_s5Plies

num_samples = 1

temp = None

loader = None
sv_filename_unlabeled_list = []
sv_orders = []
index = -1
for dir in reversed(dirs):
    index +=1
    variables = dir.split('_')
    SLLJT = variables[0]
    interaction = variables[1]
    if interaction == 'pp':
        interaction = 'pp_Vc'
    name = variables[2]
    sv_order = name[1]
    dir_path = os.path.join(directory_path, dir )
    items = os.listdir(dir_path)
    for item in items:
        item_path = os.path.join(dir_path, item)
        #print(item_path)
        print(item)
        # Check if the item is a directory
        if os.path.isdir(item_path):
            if item == 'hw_16.00':
                #print(item_path)
                for file in os.listdir(item_path):
                    print(item_path)
                    print(file)

                    file_path = os.path.join(item_path, file)

                    data_list = []
                    last_line_data_list = []


                    with open(file_path, "r") as file:
                        content = file.read()  # Die gesamte Datei als Textinhalt lesen
                        lines = content.split('\n')  # Die Zeilen anhand von Zeilenumbrüchen trennen
                        print(lines[0])
                        print(len(lines[0]))
                    #for line in lines:
                    for i in range(1,len(lines)-1):
                        line = lines[i]
                        # Die Leerzeichen als Trennzeichen verwenden, um die Elemente in jeder Zeile zu trennen
                        elements = line.split(' ')
                        numbers_list = [float(x) for x in line.split()]
                        #print(numbers_list)
                        #print(len(numbers_list))
                        #float_elements = [float(element) for element in elements]
                        data_list.append(numbers_list)
                    last_line = lines[len(lines) - 1]
                    print(last_line)
                    #last_line_values = last_line.split(' ')
                    last_line_parsed_values = [float(val) for val in last_line.split()]
                    last_line_data_list.append(last_line_parsed_values)
                    # Convert the list of lists into a NumPy array or a Pandas DataFrame
                    data_array = np.array(data_list)  # NumPy array
                    last_line_data_array = np.array(last_line_data_list)
                    if index == 0:
                        loader = np.empty((len(data_array),len(data_array[0]), 2))
                        loader[:, :, 0] = np.copy(data_array)
                        last_line_loader = np.empty((len(last_line_data_array), len(last_line_data_array[0]), 2))
                        last_line_loader[:, :, 0] = np.copy(last_line_data_array)
                    if index!= 0:
                        #loader = np.stack((loader, data_array), axis=2)
                        #last_line_loader = np.stack((last_line_loader, last_line_data_array), axis=2)
                        loader[:, :, index] = np.copy(data_array)
                        last_line_loader[:, :, index] = np.copy(last_line_data_array)
                    print(index)
                    print(np.shape(loader))
                    print(np.shape(last_line_loader))
                    sv_filename_unlabeled = f'VNN_N3LO_svPlies_SLLJT_{SLLJT}_lambda_2.00_Np_100_{interaction}_nocut.dat'
                    sv_orders.append(int(sv_order))
                    sv_filename_unlabeled_list.append(sv_filename_unlabeled)
                    '''    
                    print(np.shape(data_array))
                    print(np.shape(data_array))
                    final_array = final_array +  np.copy(data_array * singular_value)
                    print(np.shape(data_array))
                    final_last_line = final_last_line +  np.copy(last_line_data_array * singular_value)
                    '''

### have to change the singular value naming scheme to
### f'VNN_N3LO_svPlies_SLLJT_{SLLJT}_lambda_2.00_Np_100_{interaction}_nocut_s{sv_order}.dat'
### 1. change Plies -> PliesX depending on index of sample
### 2. change SVD order is now part of the name and they are all in the same foler
### ... _s{sv_order} ...
num_rows = len(data_array)
num_columns = len(data_array[0])

len_last_line = len(last_line_data_array)

higher_partial_waves_file_name = 'VNN_N3LO_Plies_lambda_2.00_hw16.00_emax_12.me2j'
higher_partial_waves_file = os.path.join(higher_partial_waves_path, higher_partial_waves_file_name)

h_data_list = []
h_last_line_data_list = []



###Load the transformed higher order partial waves

with open(higher_partial_waves_file, "r") as h_file:
    h_content = h_file.read()  # Die gesamte Datei als Textinhalt lesen
    h_lines = h_content.split('\n')  # Die Zeilen anhand von Zeilenumbrüchen trennen
for i in range(1, len(h_lines) - 1):
    h_line = h_lines[i]
    # Die Leerzeichen als Trennzeichen verwenden, um die Elemente in jeder Zeile zu trennen
    h_elements = h_line.split(' ')
    h_numbers_list = [float(h_x) for h_x in h_line.split()]
    # print(numbers_list)
    # print(len(numbers_list))
    # float_elements = [float(element) for element in elements]
    h_data_list.append(h_numbers_list)
h_last_line = h_lines[len(lines) - 1]
print('higher partial waves last line')
print(h_last_line)
# last_line_values = last_line.split(' ')
h_last_line_parsed_values = [float(h_val) for h_val in h_last_line.split()]
h_last_line_data_list.append(h_last_line_parsed_values)
# Convert the list of lists into a NumPy array or a Pandas DataFrame
h_data_array = np.array(h_data_list)  # NumPy array
h_last_line_data_array = np.array(h_last_line_data_list)
print(np.shape(h_last_line_data_array))
print(sv_orders)
print(sv_filename_unlabeled_list)
for x in range(num_samples):
    sv_filename_labeled_list = []
    singular_values_one_sample = []
    for i, sv_filename_unlabeled in enumerate(sv_filename_unlabeled_list):
        #sv_filename_labeled_list.append(sv_filename_labeled.replace("Plies", f"Plies{x}"))
        sv_filename_labeled = sv_filename_unlabeled.replace("Plies", f"Plies{x}")
        sv_path = os.path.join(singular_values_path, sv_filename_labeled)
        singular_value = np.loadtxt(sv_path)
        sv_order = sv_orders[i]
        singular_values_one_sample.append(singular_value[sv_order - 1])
        #singular_value = singular_values[int(sv_order)]  # -1 because first singular value is in the 0th row

    print(singular_values_one_sample)



###used to be both loaders[x]
    print(np.shape(singular_values_one_sample))
    summed_up = np.dot(loader, singular_values_one_sample) + h_data_array
    summed_up_last_line = np.dot(last_line_loader, singular_values_one_sample) + h_last_line_data_array
    print(h_last_line_data_array)
    print('last line shape')
    print(np.shape(summed_up_last_line))
    print(np.shape(summed_up))
    print(str(summed_up_last_line))
    print(str(summed_up_last_line[:, 0][0]))
    print(str(summed_up_last_line[:, 1][0]))
    ####ALSO ADD ALL HIGHER PARTIAL WAVES HERE

    #output_filename = 'transformed_s_p_Plies{x}.me2j'
    output_filename = f'VNN_N3LO_Plies{x}_lambda_2.00_hw16.00_emax_12.me2j'
    output_file = os.path.join(output_path, output_filename)

    with open(output_file, "w") as file:
        # write a header
        file.write('header \n')
        for i in range(num_rows):
            for j in range(num_columns):
                file.write(str(summed_up[i][j]))
                if j < num_columns - 1:
                    file.write("\t")  # Use tab as a delimiter between columns
            file.write("\n")
        print(len(summed_up_last_line[0,:]))
        for k in range(len(summed_up_last_line[0,:])):
            file.write(str(summed_up_last_line[:, k][0]))
            file.write("\t")

    #if x == 0:
     #   singular_values = singular_values_one_sample
    #else:
    #    singular_values = np.vstack(singular_values, singular_values_one_sample)




'''
num_rows = len(data_array)
num_columns = len(data_array[0])

len_last_line = len(last_line_data_array)

output_filename = 'transformed_test.me2j'
output_file = os.path.join(output_path, output_filename)

with open(output_file, "w") as file:
    #write a header
    file.write('header \n')
    for i in range(num_rows):
        for j in range(num_columns):
            file.write(str(final_array[i][j]))
            if j < num_columns - 1:
                file.write("\t")  # Use tab as a delimiter between columns
        file.write("\n")
    for k in range(len_last_line):
        file.write(last_line_data_array[k])
        file.write("\t")

'''

#temp = pd.concat([header, temp], ignore_index=True)
'''
output_filename = 'transformed_s_p.txt'
output_file = os.path.join(output_path, output_filename)

np.savetxt('results.txt', temp)
'''

