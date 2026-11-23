import paddle
import math
import numpy as np
from paddle.nn import functional as F


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = paddle.exp(paddle.arange(half_dim, dtype='float32', device=timesteps.place) * -emb)
    emb = timesteps.astype('float32')[:, None] * emb[None, :]
    emb = paddle.concat([paddle.sin(emb), paddle.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, [0, 1], pad_from_left_axis=False)
    return emb


def nonlinearity(x):
    return x * paddle.sigmoid(x)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end,
    num_diffusion_timesteps):

    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)
    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5,
            num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps,
            dtype=np.float64)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1,
            num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'sigmoid':
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas
