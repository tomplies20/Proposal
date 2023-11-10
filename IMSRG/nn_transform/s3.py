import os
import numpy as np
import pandas as pd
# Directory containing the folders
directory_path = '/data_share4/ox_tom/all_operators'  # Replace with the actual directory path
singular_values_path = '/data_share4/ox_tom/SVD_singular_values_N3LO'
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
result_accumulator = pd.DataFrame()
temp = None
for dir in reversed(dirs):
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
                    #VNN_N3LO_s5Plies_lambda_2.00_hw16.00_emax_12.me2j #transformed name
                    #VNN_N3LO_svPlies_SLLJT_11101_lambda_2.00_Np_100_nn_nocut.dat #svs name
                    sv_filename = f'VNN_N3LO_svPlies_SLLJT_{SLLJT}_lambda_2.00_Np_100_{interaction}_nocut.dat'
                    sv_path = os.path.join(singular_values_path, sv_filename)

                    singular_values = np.loadtxt(sv_path)
                    singular_value = singular_values[int(sv_order)-1] #-1 because first singular value is in the 0th row
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
                    print(np.shape(data_array))
                    print(np.shape(data_array))
                    final_array = final_array +  np.copy(data_array * singular_value)
                    print(np.shape(data_array))
                    final_last_line = final_last_line +  np.copy(last_line_data_array * singular_value)



# all operators > dirs
# 00001_nn_s1Plies > dir
## emax_20  hw_16.00  VNN_N3LO_s1Plies_SLLJT_00001_lambda_2.00_Np_100_nn_nocut.dat >items
##inside hw_16.00: VNN_N3LO_s5Plies_lambda_2.00_hw16.00_emax_12.me2j

num_rows = len(data_array)
num_columns = len(data_array[0])

#len_last_line = len(last_line_data_array)

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
    #for k in range(len_last_line):
     #   file.write(last_line_data_array[k])
     #   file.write("\t")



#temp = pd.concat([header, temp], ignore_index=True)
'''
output_filename = 'transformed_s_p.txt'
output_file = os.path.join(output_path, output_filename)

np.savetxt('results.txt', temp)
'''

