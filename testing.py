import numpy as np
import sys

def add_gaussian_noise(img, sigma, seed=None):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=img.shape)
    return img + noise

def make_debug_exponential_image(
    shape = (7000, 5000),
    amplitude = 0.042,
    decay = -0.000111,
    direction = np.deg2rad(10),
    constant = 0.0137
):
    h, w = shape
    y, x = np.mgrid[:h, :w]

    proj = np.cos(direction) * x + np.sin(direction) * y
    img = A * np.exp(decay * proj) + constant

#    img = A * compression * proj + C

    return img
