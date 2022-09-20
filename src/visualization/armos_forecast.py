import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from random import randrange
from src.models.ARMOS import ARMOS

np.set_printoptions(precision=3)

nobackup = "/net/pc160111/nobackup/users/teixeira/"

with open(nobackup + 'LoadedTrainData.pkl', 'rb') as f:
    tag, S2, var, obs = pkl.load(f)

armos1 = ARMOS(5, 1, 49)
with open('../../res/models_final/GlobalArmos1.pkl', 'rb') as f:
    params = pkl.load(f)
armos1.a, armos1.b, armos1.c, armos1.d, armos1.phi = params['a'], params['b'], params['c'], params['d'], params['phi']

loc,scale,phi = armos1.get_dist_params(var, S2)
loc,scale,phi = loc.numpy(),scale.numpy(),phi.numpy() 

i = 34

thresh = 24
t,l,s,o,p = tag[i].numpy().decode(),loc[i], scale[i], obs[i], phi[0]


l[:thresh] = l[:thresh] + p * (o[:thresh] - l[:thresh])
cum = phi**(2*np.arange(0,49))
cum = np.convolve(s[thresh:]**2, cum)[:49-thresh]
s[thresh:] = np.sqrt(cum)

plt.vlines(24,0,max(l+s), color='k', lw=1, ls='dashed')
plt.plot(l, 'b')
plt.plot(l + s, 'b--')
plt.plot(l - s, 'b--')
plt.plot(o, 'r')
plt.title(', St.'.join(t.split('_')))
plt.xlabel('Lead time (h)')
plt.ylabel('Wind speed (m/s)')
plt.savefig('test.png')
plt.show()