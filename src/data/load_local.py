import os
import numpy as np
import pickle as pkl
import tensorflow as tf
import tensorflow_probability as tfp
tfpd = tfp.distributions
from dataset import gen_dataset, load_to_memory

nobackup = '/net/pc160111/nobackup/users/teixeira/'
val_folder = nobackup + 'norm_test_data/'
train_folder = nobackup + 'norm_data/'

with open(nobackup + 'st-info.pkl', 'rb') as f:
    info = pkl.load(f)

val_files  = [val_folder + file for file in os.listdir(val_folder) if ('20181' in file or '20190' in file)]
train_files  = [train_folder + file for file in os.listdir(train_folder)]

var_list = ['PRESS', 'KINETIC', 'HUMID', 'GEOPOT']
omit = ['229', '285', '323']
sts  = [st for st in info.keys() if st not in omit]

val_dataset = gen_dataset(val_files, sts=sts, var=var_list, grid_size=5)
val_tag, val_S2, val_var, val_obs = load_to_memory(val_dataset, grid_size=5, grid_S2=True)

train_dataset = gen_dataset(train_files, sts=sts, var=var_list, grid_size=5)
tag, S2, var, obs = load_to_memory(train_dataset, grid_size=5, grid_S2=True)

with open(nobackup + 'LoadedTrainData.pkl', 'wb') as f:
    pkl.dump((tag, S2, var, obs), f)
with open(nobackup + 'LoadedValData.pkl', 'wb') as f:
    pkl.dump((val_tag, val_S2, val_var, val_obs), f)