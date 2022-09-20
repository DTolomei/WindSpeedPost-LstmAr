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
from src.models.losses import Armos1LogScore, Armos2LogScore, Armos3LogScore, ArmosL2

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

# ----------- LOSS ----------------
p = 3
LS, L2 = Armos3LogScore(49), ArmosL2(49)

# ---------- HYPERMODEL --------------
def ArmosNet(hp):
    # Hyperparameters
    P = hp.Int('p', 1,3,1)
    GridSize = hp.Int('GridSize', 5, 50, 5, default=5)
    Conv7Layers = hp.Int('Conv7Layers', 0, 4, default=0)
    Conv5Layers = hp.Int('Conv5Layers', 0, 4, default=1)
    Conv3Layers = hp.Int('Conv3Layers', 0, 4, default=1)
    LstmLayers  = hp.Int('LstmLayers' , 0, 4, default=1)
    LstmUnits   = hp.Int('LstmUnits'  , 20, 100, 5, default=25)
    LstmReg   = hp.Float('LstmReg', 0.0001, 0.01, sampling='log')
    
    # Inputs
    input_ = tfkl.Input(shape=(), dtype=tf.string)
    inputA = tfkl.Input(shape=(49, 101, 101, 1))
    inputB = tfkl.Input(shape=(49, len(var_list)))

    # Convolutional layers
    crop = tfkl.TimeDistributed(tfkl.CenterCrop(2*GridSize + 1, 2*GridSize + 1))(inputA) 
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

    # LSTM layers
    for _ in range(LstmLayers):
        x = tfkl.LSTM(
            LstmUnits,
            kernel_regularizer=tfk.regularizers.l2(LstmReg),
            return_sequences=True,
        )(x)
    
    # Peephole
    S2  = tf.math.reduce_std(crop, axis=(-3,-2))
    nwp = tfkl.Concatenate()([crop[...,GridSize,GridSize,:], inputB, S2])
    x = tfkl.TimeDistributed(tfkl.Concatenate())([x,nwp])

    
    # Output layers
    locscale = tfkl.LSTM(2, activation='softplus', return_sequences=True)(x)
    locscale = tfkl.Permute((2,1))(locscale)
    locscale = tfkl.Flatten()(locscale)
    phi = tfkl.TimeDistributed(tfkl.Dense(P, activation='relu'))(x)
    phi = tfkl.Flatten()(phi)
    phi = tfkl.Dense(P, activation='sigmoid')(phi)
    out = tfkl.Concatenate()([locscale, phi])

    model = tfk.models.Model(inputs=[input_,inputA, inputB], outputs=out) 

    config = model.get_config()
    custom_objects = {"LS": LS, "ArmosL2":ArmosL2}
    with tfk.utils.custom_object_scope(custom_objects):
        model = tfk.Model.from_config(config)
        
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.001, clipvalue=1.0), 
        loss=LS,
        metrics=[ArmosL2]
    )
    
    return model

with open(home+f'res/Armos{p}NetHp.pkl','rb') as f:
    hp = pkl.load(f)
    
print(f'Train Armos{p}Net')
for i in range(3):
    model = ArmosNet(hp)
    
    hist = model.fit(
            train_dataset,
            epochs=150,
            callbacks=[
                    tf.keras.callbacks.ModelCheckpoint(home + f'res/ckpts/Armos{p}NetTrial{i}.h5', monitor='loss', save_best_only=True),
                ],
            )
