import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfpd = tfp.distributions
import datetime as dt
from scipy import optimize
import pickle as pkl

class SchaakeEMOS:
    def __init__(self, num_var, T=49):
        self.a = tf.Variable(tf.zeros(T))
        self.b = tf.Variable(tf.zeros((T,num_var)))
        self.c = tf.Variable(tf.ones(T))
        self.d = tf.Variable(tf.zeros(T))
        self.opt = tf.keras.optimizers.SGD(learning_rate=1)
    
    def get_params(self):
        return self.a, self.b, self.c, self.d
    
    def forecast_dist(self, x, S2):
        loc = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
#         scale = self.c + self.d * S2
        scale = tf.sqrt(self.c + self.d * S2)
        dist = tfpd.TruncatedNormal(loc, scale, low=0.0, high=10000000)
        return dist
    
    def loss(self, x, S2, obs):
        dist = self.forecast_dist(x, S2)
        return - dist.log_prob(obs)
    
    def fit(self, x, S2, obs, steps=1000):
        hist = []
        for _ in range(steps):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(self.loss(x, S2, obs))
            step = self.opt.minimize(loss, [self.a,self.b,self.c,self.d], tape=tape)
            hist.append(loss.numpy())
            print('Step', step.numpy(),': loss',loss.numpy(), end='\r')
        return hist
    
    def _isclose(self, date, target, tolerance):
        target_y, target_m = target.year, target.month
        date_m, date_d = date.month, date.day
        if date_m >= 6 and target_m <= 6:
            target_y -= 1
        if date_m <= 6 and target_m >= 6:
            target_y += 1
        if date_m == 2 and date_d == 29:
            date_d = 28
        new_date = dt.datetime(target_y, date_m, date_d)
        return abs((new_date - target).days) <= tolerance
    
    def schaake_shuffle(self, tag, x, S2, past_tag, past_obs, n_samples=20, random=False):
        dist = self.forecast_dist(x, S2)
        samples = dist.sample(n_samples).numpy()
        samples.sort(0)
        
        # Get indices of similar dates
        idx = np.random.choice(past_obs.shape[0], samples[...,0].size, replace=True)
        idx = idx.reshape(samples[...,0].shape)
        if not random:
            tag= tag.numpy().astype('str')
            past_tag = [(dt.datetime.strptime(t[:8], "%Y%m%d"), t[-3:]) for t in past_tag.numpy().astype('str')]
            idxs = []
            for t in tag:
                target,  st = dt.datetime.strptime(t[:8], "%Y%m%d"), t[-3:]
                idx, tol = [], (n_samples - 1.0) / 2.0
                while len(idx) < n_samples:
                    idx = [i for i, (date, s) in enumerate(past_tag) if s==st and self._isclose(date, target, tol)]
                    tol += 1
                idxs.append(idx[:n_samples]) 
            idx = np.array(idxs, dtype='int').T

        # Get shuff template
        y = past_obs.numpy()[idx]
        perm = y.argsort(0)

        # Shuffle!
        shuff = np.zeros(samples.shape)
        idx = np.indices(samples.shape)
        idx[0] = perm
        idx = tuple(np.split(idx, idx.shape[0], axis=0))
        shuff[idx] = samples
        return shuff