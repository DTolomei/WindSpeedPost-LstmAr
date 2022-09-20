import numpy as np
import tensorflow as tf
tfk  = tf.keras
tfkl = tfk.layers
import tensorflow_probability as tfp
tfpd = tfp.distributions

EPS = 0.1
WEIGHTS = tf.constant([[2**(1 - abs(i - j)) for i in range(49)] for j in range(49)])

def EmosLogScore(T):
    def LS(obs, out):
        loc, scale = out[...,0], out[...,1]
        dist = tfpd.TruncatedNormal(loc, scale + EPS, 0.0, 1000000.0)
        return - dist.log_prob(obs)
    return LS

def EmosEnergyScore(T, p=1.0, n_samples=30):
    def ES(obs, out):
        loc, scale = out[...,0], out[...,1]
        dist = tfpd.TruncatedNormal(loc, scale + EPS, 0.0, 1000000.0)
        X1 = dist.sample(n_samples) #(n_samples, batch, T)
        X2 = dist.sample(n_samples) #(n_samples, batch, T)
        E1 = tf.norm(X1 - obs, axis=-1)**p
        E2 = tf.norm(X1 - X2 , axis=-1)**p
        E1 = tf.reduce_mean(E1, axis=0)
        E2 = tf.reduce_mean(E2, axis=0)
        return E1 - 0.5 * E2
    return ES

def EmosVariogramScore(T, p=0.5, w=WEIGHTS, n_samples=20):
    def VS(obs, out):
        loc, scale = out[...,0], out[...,1]
        dist = tfpd.TruncatedNormal(loc, scale + EPS, 0.0, 1000000.0)
        X = dist.sample(n_samples) #(n_samples, batch, T)
        D1 = tf.math.abs(obs[...,None] - obs[...,None,:])**p #(batch, T, T)
        D2 = tf.math.abs(X[...,None] - X[...,None,:])**p #(n_samples, batch, T, T)
        VS = (D1 - tf.reduce_mean(D2, axis=0))**2
        VS  = w * tf.linalg.band_part(VS, 0, -1)
        return tf.reduce_sum(VS, axis=(-2,-1))    
    return VS

def Armos1LogScore(T):    
    def LS(obs, out):
        loc   = out[...,:T]
        scale = out[...,T:-1]
        phi   = out[...,-1:]
        res = obs - loc
        res = phi * res[...,:-1]
        res = tf.concat([tf.zeros((len(res),1)), res], axis=-1)
        loc  = loc + res
        dist = tfpd.TruncatedNormal(loc, scale + EPS, 0.0, 1000000.0)
        return - dist.log_prob(obs)
    return LS

def Armos2LogScore(T):    
    def LS(obs, out):
        loc   = out[...,:T]
        scale = out[...,T:-2]
        phi   = out[...,-2:]
        res = obs - loc
        res1 = phi[...,:1] * res[...,1:T-1]
        res2 = phi[...,1:] * res[...,:T-2]
        res = res1 + res2
        res = tf.concat([tf.zeros((len(res),2)), res], axis=-1)
        loc  = loc + res
        dist = tfpd.TruncatedNormal(loc, scale + EPS, 0.0, 1000000.0)
        return - dist.log_prob(obs)
    return LS

def Armos3LogScore(T):    
    def LS(obs, out):
        loc   = out[...,:T]
        scale = out[...,T:-3]
        phi   = out[...,-3:]
        res = obs - loc
        res1 = phi[..., :1] * res[...,2:T-1]
        res2 = phi[...,1:2] * res[...,1:T-2]
        res3 = phi[...,2:3] * res[..., :T-3]
        res = res1 + res2 + res3
        res = tf.concat([tf.zeros((len(res),3)), res], axis=-1)
        loc  = loc + res
        dist = tfpd.TruncatedNormal(loc, scale + EPS, 0.0, 1000000.0)
        return - dist.log_prob(obs)
    return LS

def ArmosL2(T):
    def L2(obs, out):
        return tf.reduce_mean(tf.square(obs - out[...,:T]))
    return L2

# ----------------------- OLD ------------------------------
# log_score = lambda y, dist : - dist.log_prob(y)

# def _mean_p_norm(samples, p=1.0):
#     norm   = tf.cast(tf.norm(samples, axis=-1), 'double')
#     norm_p = tf.pow(norm, tf.broadcast_to(tf.cast(p, 'double'), tf.shape(norm)))
#     return tf.reduce_mean(norm_p, axis=0)

# def _pairwise_p_diff(p, x):
#     diff = tf.abs(x[..., tf.newaxis] - x[..., tf.newaxis, :])
#     return tf.pow(tf.cast(diff, 'double'), tf.broadcast_to(tf.cast(p, 'double'), tf.shape(diff)))

# def EnergyScore(n_samples=100, p = 1.0):
#     def score(y, dist):
#         X1 = dist.sample(n_samples)
#         X2 = dist.sample(n_samples)
#         E1 = _mean_p_norm(X1 - y , p)
#         E2 = _mean_p_norm(X1 - X2, p)
#         return tf.add(E1, tf.scalar_mul(- 0.5, E2))
#     return score

# def VariogramScore(n_samples=100, weights=[1.0], p=1.0):
#     def score(y, dist):
#         X = dist.sample(n_samples)
#         X_diff_p = _pairwise_p_diff(p, X)
#         Y_diff_p = _pairwise_p_diff(p, y)
#         Z = tf.square(Y_diff_p - tf.reduce_mean(X_diff_p, axis=0))
#         Z = tf.linalg.band_part(Z, 1, -1)
#         return tf.reduce_sum(weights * Z, axis=(-2, -1))
#     return score


# def batch_convolve(tup):
#     res, phi = tup
#     return tf.nn.conv1d(
#                 res[None,:,None], 
#                 phi[::-1, None, None],
#                 stride=1,
#                 padding='VALID'
#             )[0,:-1,0]

# def ArmosLogScore(T, p):    
#     def score(y, out):
#         loc   = out[...,:T]
#         scale = out[...,T:2*T]
#         phi  = out[...,-p:]
#         res  = y - loc
#         conv = tf.map_fn(
#                 fn= batch_convolve,
#                 elems=(res, phi),
#                 fn_output_signature=tf.float32
#             )
#         conv = tf.concat([tf.zeros((len(conv),p), dtype=tf.float32), conv], 1)
#         loc  = loc + conv
#         dist  = tfpd.TruncatedNormal(loc,scale,0,100000)
#         return - tf.reduce_mean(dist.log_prob(y))
#     return score

# # def ArmosLogScore(T, p):    
# #     const = tf.cast(tf.math.log(2 * np.pi) * T / 2.0, tf.float32)
# #     def score(y, out):
# #         loc   = out[...,:T]
# #         scale = out[...,T:2*T]
# #         phis  = out[...,-p:]
# #         log_scale = tf.reduce_sum(tf.math.log(scale), axis=-1)
# #         res  = tf.math.subtract(y,loc)
# #         conv = tf.map_fn(
# #                 fn= batch_convolve,
# #                 elems=(res, phis),
# #                 fn_output_signature=tf.float32
# #             )
# #         conv = tf.concat([tf.zeros((len(conv),p), dtype=tf.float32), conv], 1)
# #         res = tf.math.subtract(res, conv)
# #         res = tf.math.divide(res, scale)
# #         res = tf.reduce_sum(tf.math.square(res), axis=-1)
# #         return const + tf.reduce_mean(log_scale + res)
# #     return score

# def ArmosL2Penalty(T, p):
#     LS = ArmosLogScore(T,p)
#     def L2(obs, out):
#         return LS(obs,out) + tf.reduce_mean(tf.square(obs - out[...,:T])) / 100.0
#     return L2

