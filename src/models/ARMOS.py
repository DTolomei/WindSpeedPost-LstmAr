import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfpd = tfp.distributions
from scipy import optimize
import pickle as pkl

    
class ARMOS:
    def __init__(self, num_var, p, T=49):
        self.T   = T
        self.p   = p
        self.a   = tf.Variable(tf.zeros(T))
        self.b   = tf.Variable(tf.zeros((T,num_var)))
        self.c   = tf.Variable(tf.ones(T))
        self.d   = tf.Variable(tf.zeros(T))
        self.phi = tf.Variable(tf.zeros(p))
        self.opt = tf.keras.optimizers.SGD(learning_rate=1, clipvalue=0.1)
        
    def get_params(self):
        return self.a, self.b, self.c, self.d, self.phi
    
    def get_dist_params(self, x, S2):
        loc   = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)
#         scale = self.c + self.d * S2
        return loc, scale, self.phi
    
    def loss(self, x, S2, obs):
        loc   = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)
#         scale = self.c + self.d * S2
        res   = obs - loc
        corr  = tf.zeros(res.shape[:-1] + (self.T - self.p,))
        for i in range(self.p):
            corr = corr + self.phi[i] * res[...,i:self.T+i-self.p]
        corr  = tf.concat([tf.zeros(res.shape[:-1] + (self.p,)), corr], -1)
        loc   = loc + corr
        dist  = tfpd.TruncatedNormal(loc,scale,0, 100000)
        return - dist.log_prob(obs)
    
    def fit(self, x, S2, obs, steps=1000):
        hist = []
        for _ in range(steps):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(self.loss(x, S2, obs))
            step = self.opt.minimize(loss, [self.a,self.b,self.c,self.d,self.phi], tape=tape)
            hist.append(loss.numpy())
            print('Step', step.numpy(),': loss',loss.numpy(), end='\r')
        return hist
    
    def sample(self, x, S2, n_samples=100):
        loc   = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)
#         scale = self.c + self.d * S2
        samples = np.zeros((n_samples,) + loc.shape)
        dist = tfpd.TruncatedNormal(loc[...,:self.p], scale[...,:self.p], 0, 10000000)
        samples[...,:self.p] = dist.sample(n_samples).numpy()
        for t in range(self.p, self.T):
            res  = samples[...,t-self.p:t] - loc[...,t-self.p:t]
            corr = tf.reduce_sum(res * self.phi, axis=-1)
            dist = tfpd.TruncatedNormal(loc[...,t] + corr, scale[...,t], 0, 10000000)
            samples[...,t] = dist.sample(1).numpy()
        return samples