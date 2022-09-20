import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfpd = tfp.distributions
import datetime as dt

EPS = 0.1
WEIGHTS = np.array([[2**(1 - abs(i - j)) for i in range(49)] for j in range(49)])

def isclose(date, target, tolerance):
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

def schaake_shuffle(tag, samples, past_tag, past_obs, random=False):
    n_samples = samples.shape[0]
    sort = np.sort(samples, axis=0)
    
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
                idx = [i for i, (date, s) in enumerate(past_tag) if s==st and isclose(date, target, tol)]
                tol += 1
            idxs.append(idx[:n_samples]) 
        idx = np.array(idxs, dtype='int').T
            
    y = np.array(past_obs)[idx]
    perm = y.argsort(0)

    shuff = np.zeros(sort.shape)
    idx = np.indices(sort.shape)
    idx[0] = perm
    idx = tuple(np.split(idx, idx.shape[0], axis=0))
    shuff[idx] = sort
    return shuff

# Energy Score
def energy_score(samples, obs):
    s1, s2 = np.split(samples, 2, axis=0)
    ES = np.linalg.norm(s1 - obs, axis=-1) - 0.5 * np.linalg.norm(s1 - s2, axis=-1)
    return ES.mean(0)

# Variogram Score
def variogram_score(samples, obs, p=0.5, w=WEIGHTS):
    VS  =  np.abs(obs[..., None] - obs[..., None, :])**p
    VS -= (np.abs(samples[..., None] - samples[..., None, :])**p).mean(axis=0)
    VS  = w * np.triu(VS**2)
    VS  = VS.sum(axis=(-2,-1))    
    return VS

def EmosSample(loc, scale, n_samples=30):
    dist = tfpd.TruncatedNormal(loc, scale + EPS, 0.0, 1000000.0)
    return dist.sample(n_samples).numpy()

def ArmosSample(loc, scale, phi, n_samples=30):
    batch, p = phi.shape
    samples = np.zeros((n_samples,) + loc.shape)
    dist = tfpd.TruncatedNormal(loc[...,:p], scale[...,:p] + EPS, 0.0, 1000000.0)
    samples[...,:p] = dist.sample(n_samples).numpy()
    for t in range(p, 49):
        res  = samples[...,t-p:t] - loc[...,t-p:t]
        corr = tf.reduce_sum(res * phi[::-1], axis=-1)
        dist = tfpd.TruncatedNormal(loc[...,t] + corr, scale[...,t] + EPS, 0.0, 1000000.0)
        samples[...,t] = dist.sample(1).numpy()
    return samples





def get_past_obs(dataset):
    past_obs = []
    for i, (_, y) in dataset.enumerate():
        print('Get past obs:',i.numpy(), end='\r')
        past_obs.append(y.numpy())
    return np.concatenate(past_obs, axis=0)


# def ARMOS_sample(loc, scale, phi, n_samples=100):
#     batch, p = phi.shape
#     samples = np.zeros((n_samples,) + loc.shape)
#     samples[...,:p] = norm.rvs(loc=loc[:,:p], scale=scale[:,:p], size=(n_samples, batch, p))
#     for t in range(p, loc.shape[-1]):
#         dev = samples[...,t-p:t] - loc[...,t-p:t]
#         update = (dev * phi[...,::-1]).sum(-1)
#         samples[...,t] = norm.rvs(loc=loc[:,t] + update, scale=scale[:,t], size=(n_samples, batch))
#     return samples

def get_ARMOS_samples(model, p, dataset, n_samples=100):
    obs, samples = [], []
    for i, (x, y) in dataset.enumerate():
        print('Get pred:', i.numpy(), end='\r')
        s = model(x).numpy()
        loc = s[:,:49]
        scale = s[:,:49:-p]
        phi = s[:,-p:]
        samples.append(ARMOS_sample(loc, scale, phi, n_samples))
        obs.append(y.numpy())
    samples = np.concatenate(samples, axis=1)
    obs     = np.concatenate(obs, axis=0)
    return samples, obs

def get_dist_samples(model, dataset, n_samples=100):
    obs, samples = [], []
    for i, (x, y) in dataset.enumerate():
        print('Get pred:', i.numpy(), end='\r')
        s = model(x).sample(n_samples).numpy()
        samples.append(s)
        obs.append(y.numpy())
    samples = np.concatenate(samples, axis=1)
    obs     = np.concatenate(obs, axis=0)
    return samples, obs

# Schaake Shuffle
def _isclose(date, target, tolerance):
    target_y, target_m = target.year, target.month
    date_m, date_d = date.month, date.day
    if date_m >= 6 and target_m <= 6:
        target_y -= 1
    if date_m <= 6 and target_m >= 6:
        target_y += 1
    new_date = dt.datetime(target_y, date_m, date_d)
    return abs((new_date - target).days) <= tolerance
