import os
import pickle as pkl
import numpy as np
from tqdm import tqdm
from dataset import gen_dataset

nobackup = '/net/pc160111/nobackup/users/teixeira/'

with open(home + 'st-info.pkl', 'rb') as f:
    info = pkl.load(f)
    
# Get files
data_folder = home + 'clean_data/'
files = [data_folder + file for file in os.listdir(data_folder)]

# Select stations and variables
omit = ['229', '285', '323']
var_list = ['PRESS', 'KINETIC', 'HUMID', 'GEOPOT']

# Build datasets
sts  = [st for st in info.keys() if st not in omit]

dataset = gen_dataset(files, sts=sts, var=var_list, grid_size=1)
_, _, variables, _ = load_to_memory(dataset, grid_size=1, grid_S2=True)

mean, std = variables.mean(axis=(0,1)), var.std(axis=(0,1))

for file in os.listdir(clean_folder)[1:]:
    with open(data_folder + file, 'rb') as f:
        data = pkl.load(f)
    for key in data.keys():
        if not key == 'GRID':
            data[key]['PRESS'] -= mean[0]
            data[key]['PRESS'] /= std[0]
            data[key]['KINETIC'] -= mean[1]
            data[key]['KINETIC'] /= std[1]
            data[key]['HUMID'] -= mean[2]
            data[key]['HUMID'] /= std[2]
            data[key]['GEOPOT'] -= mean[3]
            data[key]['GEOPOT'] /= std[3]
    with open(clean_folder + file,'wb') as f:
        pkl.dump(data,f)