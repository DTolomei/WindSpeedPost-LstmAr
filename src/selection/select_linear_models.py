import os
import numpy as np
import pickle as pkl
import tensorflow as tf
import tensorflow_probability as tfp
tfpd = tfp.distributions
from src.data.dataset import gen_dataset, load_to_memory
from src.models.SchaakeEMOS import SchaakeEmos
from tqdm import tqdm


opt_grids = {}
for st in sts:
    try:
        print(st)
        # Load st data
        val_dataset   = gen_dataset(val_files  , sts=[st], var=var_list, grid_size=grid_size)
        train_dataset = gen_dataset(train_files, sts=[st], var=var_list, grid_size=grid_size)
        val_tag, val_grid, val_var, val_obs = load_to_memory(val_dataset, grid_size)
        tag, grid, var, obs = load_to_memory(train_dataset, grid_size)
        
        # Select grid size
        sizes = {}
        for S2_size in tqdm(range(0, 50, 5)):
            val_S2  = val_grid[...,grid_size-S2_size:grid_size+S2_size+1,grid_size-S2_size:grid_size+S2_size+1]
            val_S2  = tf.math.reduce_std(val_S2, axis=(-2,-1))
            S2  = grid[...,grid_size-S2_size:grid_size+S2_size+1,grid_size-S2_size:grid_size+S2_size+1]
            S2  = tf.math.reduce_std(S2, axis=(-2,-1))
            
            emos = SchaakeEMOS(1)
            hist = emos.fit(var[...,:1], S2, obs)
            sizes[S2_size] = emos.loss(val_var[...,:1], val_S2, val_obs).numpy()
        opt_grids[st] = min(sizes,key=sizes.get)
    except:
        print('Error', st)
with open(nobackup + 'opt_grid_sizes.pkl','wb') as f:
    pkl.dump(opt_grids, f)
    
forwards = {}
S2_size = 3

for st in sts:
    try:
        print('Selection for', st)
        val_dataset   = gen_dataset(val_files  , sts=st, var=var_list, grid_size=grid_size)
        train_dataset = gen_dataset(train_files, sts=st, var=var_list, grid_size=grid_size)

        val_tag, val_grid, val_var, val_obs = load_to_memory(val_dataset, grid_size)
        tag, grid, var, obs = load_to_memory(train_dataset, grid_size)

        # Spatial variance hyperparameter
        val_S2  = val_grid[...,grid_size-S2_size:grid_size+S2_size+1,grid_size-S2_size:grid_size+S2_size+1]
        val_S2  = tf.math.reduce_std(val_S2, axis=(-2,-1))
        S2  = grid[...,grid_size-S2_size:grid_size+S2_size+1,grid_size-S2_size:grid_size+S2_size+1]
        S2  = tf.math.reduce_std(S2, axis=(-2,-1))

        model = SchaakeEMOS(1, 49)
        print("Base ['WIND10']")
        hist  = model.fit(var[...,:1], S2, obs, steps=5000)
        score = model.validate(val_var[...,:1], val_S2, val_obs, obs)
        print('Score :', score['ES'])

        forward = [(0,'WIND10',score)]
        var_index = [(i+1, var_name) for i, var_name in enumerate(var_list)]

        while var_index:
            tests = []
            for i, var_name in var_index:
                print('Testing', [k for _,k,_ in forward] + [var_name])
                idx = [j for j,_,_ in forward] + [i]
                model = SchaakeEMOS(len(forward)+1, 49)
                hist  = model.fit(tf.gather(var, idx, axis=-1), S2, obs, steps=5000)
                score = model.validate(tf.gather(val_var, idx, axis=-1), val_S2, val_obs, obs)
                tests.append((i, var_name, score))
                print('Score :', score['ES'])
            best_var = min(tests, key=lambda x:x[-1]['ES'])
            forward.append(best_var)
            var_index.remove(tuple(best_var[:-1]))
        forwards[st] = forward
    except:
        print('Error for', st, '!')