from astropy.io import fits
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation
from scipy.ndimage import gaussian_filter
import argparse
import configparser
import sys
from pathlib import Path

def add_gaussian_noise(img, sigma, seed=None):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=img.shape)
    return img + noise

def make_debug_exponential_image(
    shape=(7000, 5000),
    A=0.042,
    compression=-0.000111,
    angle=np.deg2rad(10),
    C=0.0137
):
    h, w = shape
    y, x = np.mgrid[:h, :w]

    proj = np.cos(angle) * x + np.sin(angle) * y
    img = A * np.exp(compression * proj) + C

#    img = A * compression * proj + C

    return img
