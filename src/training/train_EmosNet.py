import os
import pickle as pkl
import tensorflow as tf
tfk  = tf.keras
tfkl = tfk.layers
import tensorflow_probability as tfp
tfpd = tfp.distributions
tfpl = tfp.layers
import keras_tuner as kt
from tensorflow.keras.utils import plot_model
# Custom stuff
from src.data.dataset import gen_dataset
from src.models.losses import EmosLogScore, EmosEnergyScore, EmosVariogramScore

home = '/home/dtst/'

# Get files
with open(home + 'st-info.pkl', 'rb') as f:
    info = pkl.load(f)
    
data_folder = home + 'data/'
train_files = [data_folder + file for file in os.listdir(data_folder)]

# Select stations and variables
omit = ['229', '285', '323']
var_list = ['PRESS', 'KINETIC', 'HUMID', 'GEOPOT']

# Datasets
sts  = [st for st in info.keys() if st not in omit]
train_dataset = gen_dataset(train_files, sts=sts, var=var_list, grid_size=50).batch(16).shuffle(6, reshuffle_each_iteration=True)

# ---------- LOSSES --------------
LS, ES, VS = EmosLogScore(49), EmosEnergyScore(49), EmosVariogramScore(49)

# ---------- HYPERMODEL --------------
def EmosNet(hp):
    # Hyperparameters
    GridSize = hp.Int('GridSize', 5, 50, 5, default=5)
    Conv7Layers = hp.Int('Conv7Layers', 0, 4, default=0)
    Conv5Layers = hp.Int('Conv5Layers', 0, 4, default=1)
    Conv3Layers = hp.Int('Conv3Layers', 0, 4, default=1)
    DenseLayers  = hp.Int('DenseLayers', 0, 4, default=1)
    DenseUnits   = hp.Int('DenseUnits', 20, 100, 5, default=25)
    DenseReg   = hp.Float('DenseReg', 0.0001, 0.01, sampling='log')
    
    # Inputs
    input_ = tfkl.Input(shape=(), dtype=tf.string)
    inputA = tfkl.Input(shape=(49, 101, 101, 1))
    inputB = tfkl.Input(shape=(49, len(var_list)))

    # Convolutional layers
    crop = tfkl.TimeDistributed(tfkl.CenterCrop(2*GridSize+1, 2*GridSize+1))(inputA) 
    x = crop
    for _ in range(Conv7Layers):
        x = tfkl.TimeDistributed(
                tfkl.Conv2D(
                    filters=8,
                    kernel_size=5,
                    padding='same',
                    activation='relu'
                )
            )(x)
        x = tfkl.BatchNormalization()(x)
    x = tfkl.TimeDistributed(tfkl.MaxPooling2D())(x)
    for _ in range(Conv5Layers):
        x = tfkl.TimeDistributed(
                tfkl.Conv2D(
                    filters=16,
                    kernel_size=5,
                    padding='same',
                    activation='relu'
                )
            )(x)
        x = tfkl.BatchNormalization()(x)
    x = tfkl.TimeDistributed(tfkl.MaxPooling2D())(x)
    for _ in range(Conv3Layers): 
        x = tfkl.TimeDistributed(
                tfkl.Conv2D(
                    filters=32,
                    kernel_size=5,
                    padding='same',
                    activation='relu'
                )
            )(x)
        x = tfkl.BatchNormalization()(x)
    x = tfkl.TimeDistributed(tfkl.MaxPooling2D())(x)
    x = tfkl.TimeDistributed(tfkl.Flatten())(x)

    # Dense layers
    for _ in range(DenseLayers):
        x = tfkl.TimeDistributed(tfkl.Dense(
                DenseUnits,
                kernel_regularizer=tfk.regularizers.l2(DenseReg),
                activation='relu',
            ))(x)
    x = tfkl.TimeDistributed(tfkl.Dense(
          units=20, 
          activation='relu',
      ))(x)

    # Peephole
    S2  = tf.math.reduce_std(crop, axis=(-3,-2))
    nwp = tfkl.Concatenate()([crop[...,GridSize,GridSize,:], inputB, S2])
    x = tfkl.TimeDistributed(tfkl.Concatenate())([x,nwp])
    x = tfkl.Flatten()(x)
    
    # Output layers
    x = tfkl.Dense(2 * 49, activation='softplus')(x)
    out = tfkl.Reshape((49,2))(x)
    
    model = tfk.models.Model(inputs=[input_,inputA, inputB], outputs=out) 

    config = model.get_config()
    custom_objects = {"LS": LS, "ES":ES, "VS":VS}
    with tfk.utils.custom_object_scope(custom_objects):
        model = tfk.Model.from_config(config)
        
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.001, clipvalue=1.0), 
        loss=LS,
        metrics=[ES,VS]
    )
    
    return model

with open(home+'res/EmosNetHp.pkl','rb') as f:
    hp = pkl.load(f)

print('Train LstmEmosNet')
for i in range(3):
    model = EmosNet(hp)
    
    hist = model.fit(
            train_dataset,
            epochs=150,
            callbacks=[
                    tf.keras.callbacks.ModelCheckpoint(home + f'res/ckpts/EmosNetTrial{i}.h5', monitor='loss', save_best_only=True),
                ],
            )