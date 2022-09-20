import os
import numpy as np
import pickle as pkl
import tensorflow as tf
from random import shuffle as RandomShuffle

def gen_dataset(files, sts=[], var=[], grid_size=10, cycle_length=3, block_length=2, shuffle=True):
    """Generates a TensorFlow dataset object from Pickle files.

    Parameters
    ----------
    files : List[str]
        List of file paths for the Pickle files from which the dataset is generated.
    sts : List[str]
        List of station included in the dataset, each represented by its code.
    var : List[str]
        List of weather variables included in the dataset.
    grid_size : int
        Size of the grid of wind speed forecasts surrounding the station.
        Grid has dimensions (2 * grid_size + 1) X (2 * grid_size + 1).
    cycle_length : int
        Number of interlieved files for output in final dataset.
    block_length : int
        Number of samples per file during interleaving.
    shuffle : bool
        Shuffle files.
        
    Returns
    -------
    tf.data.Dataset
        Dataset used for model training/validation/testing.
    """
    
    # Generate file dataset
    if shuffle:
        RandomShuffle(files)
    file_dataset = tf.data.Dataset.from_tensor_slices(files)

    # Generator for individual samples
    # Cuts the grid centered at each station location.
    def cut_grid(file_path):
        file_path_str = file_path.decode()
        with open(file_path_str, 'rb') as f:
            data = pkl.load(f)
        grid = data.pop('GRID')
        for tag, local in data.items():
            if tag[-3:] in sts:
                obs = local.pop('OBS')
                loci, locj = local.pop('LOC')
                loc_grid = grid[:, loci-grid_size:loci+grid_size+1, locj-grid_size:locj+grid_size+1, None]
#                 loc_var  = np.asarray([x for _, x in sorted(local.items())]).T
                loc_var = np.asarray([local[key] for key in var]).T
                yield (tf.constant(tag), loc_grid, loc_var), obs

    # Builds dataset from generator
    # Reads data sequentially from especific file
    def get_samples(file_path): 
        return tf.data.Dataset.from_generator(
            cut_grid, 
            args=(file_path,),
            output_signature=((
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(49, 2*grid_size+1, 2*grid_size+1, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(49, len(var)), dtype=tf.float32)),
                tf.TensorSpec(shape=(49), dtype=tf.float32)
            )
        )
    
    # Return interleaved dataset
    return file_dataset.interleave(
        get_samples, 
        cycle_length=cycle_length, 
        block_length=block_length, 
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)




def load_to_memory(dataset, grid_size, grid_S2=False, add_wind10=True):
    """Loads TensorFlow dataset to memory.

    Parameters
    ----------
    dataset : tf.data.Dataset
        TensorFlow dataset to be loaded to memory.
    grid_size : int
        Grid size to be read, possibly smaller than the original dataset grid_size.
    grid_S2 : bool
        Calculate the variance of the grid of wind speeds as an extra variable.
    add_wind10 : bool
        Add wind speed at 10 m at the central grid point as an extra variable.
        
    Returns
    -------
    tag : tf.Tensor
        Tensor with tags (date/location).
    grid : tf.Tensor
        Tensor with grids of wind speed forecasts.
    var : tf.Tensor
        Tensor with auxiliary variables at central grid point.
    obs : tf.Tensor
        Tensor with wind speed observations.
    """
    
    tag, grid, var, obs = [], [], [], []
    count = 0
    for (t, g, v), o in dataset:
        count += 1
        print('Sample', count, end='\r')
        tag.append(t)
        if not grid_S2:
            grid.append(g[...,0])
        else:
            grid.append(tf.math.reduce_variance(g[...,0], axis=(-2,-1)))
        if add_wind10:
            w = g[...,grid_size,grid_size,:]
            v = tf.concat([w,v], axis=-1)
        var.append(v)
        obs.append(o)
    tag  = tf.stack(tag , axis=0)
    grid = tf.stack(grid, axis=0)
    var  = tf.stack(var , axis=0)
    obs  = tf.stack(obs , axis=0)
    return tag, grid, var, obs

