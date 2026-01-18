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

class exponential_model:
	def __init__(self, img, mask, settings):
		self.initial_amplitude_clip_percentiles = settings["inital_amplitude_clip_percentile_min"], settings["inital_amplitude_clip_percentile_max"]
		self.inital_significant_gradient_min_percentile = settings["inital_significant_gradient_min_percentile"]
		self.params = self.estimate_initial_values(img, mask)
    
    # Calculates the lentgh of the projected gradient vector for each pixel.
    def directional_gradient(img, angle, mask):
        dIy, dIx = np.gradient(img)
        grad_dir = np.cos(angle) * dIx + np.sin(angle) * dIy
        return grad_dir[mask]
    
    def calc_diagonal(img):
        diagonal_x, diagonal_y = img.shape
        return (diagonal_x ** 2.0 + diagonal_y ** 2.0) ** 0.5
    
    # Calculates the distance of points from a line which goes through the origin.
    # Returns a 2D array.
    def project_coordinates(shape, angle):
        height, width = shape
        y, x = np.indices((height, width))
        projected_coordinates = np.cos(angle) * x + np.sin(angle) * y
        return projected_coordinates
    
    def estimate_decay_and_constant(self, projected_vals, gradient, img_flattened):
	    valid_values_mask = np.isfinite(projected_vals) & np.isfinite(gradient) & np.isfinite(img_flattened)
	    valid_gradient = gradient[valid_values_mask]
	    valid_values = img_flattened[valid_values_mask]
	
	    valid_gradient_absolute = np.abs(valid_gradient)
	    valid_significant_gradient = valid_gradient_absolute > np.percentile(valid_gradient_absolute, self.inital_significant_gradient_min_percentile)
	
	    valid_gradient = valid_gradient[valid_significant_gradient]
	    valid_values = valid_values[valid_significant_gradient]
	
	    tmp = np.vstack([valid_values, np.ones_like(valid_values)]).T
	    decay, intercept = np.linalg.lstsq(tmp, valid_gradient, rcond=None)[0]
	
	    constant = -intercept / decay
	
	    return decay, constant
    
    def estimate_initial_values(self, img, mask):
	    angle = self.estimate_angle(img, mask)
	
	    projected_vals = project_coordinates(img.shape, angle)[mask]
	    img_vals = img[mask]
	
	    gradient = directional_gradient(img, angle, mask)
	    decay, constant = estimate_decay_and_constant(projected_vals, gradient, img_vals)
	    
	    return estimate_amplitude(img, mask, angle, decay, constant, self.initial_amplitude_clip_percentiles), decay * calc_diagonal(img), angle, C

    def background_model(self, x, y):
        coefficient, decay, angle, constant = self.params
        proj = np.cos(angle) * x + np.sin(angle) * y
        return coefficient * np.exp(decay * proj) + constant

    def estimate_angle(self, img, mask):
        y, x = np.nonzero(mask)
        z = img[y, x]

        X = np.column_stack([
            x,
            y,
            np.ones_like(x)
        ])

        (a, b, c), *_ = np.linalg.lstsq(X, z, rcond=None)

        return np.arctan2(b, a)
    
    def estimate_amplitude(img, mask, angle, decay, constant, clip_percentiles):
	    height, width = img.shape
	    y, x = np.mgrid[:height, :width]
	
	    projected_coordinates = np.cos(angle) * x + np.sin(angle) * y
	    exponentials = np.exp(decay * projected_coordinates)
	
	    masked_img = img[mask]
	    masked_exponentials = exponentials[mask]
	
	    valid_mask = np.isfinite(masked_exponentials) & (masked_exponentials > 0)
	    masked_img = masked_img[valid_mask]
	    masked_exponentials = masked_exponentials[valid_mask]
	
	    ratio = (masked_img - constant) / masked_exponentials
	    ratio = ratio[np.isfinite(ratio)]
	    
	    lo, hi = np.percentile(ratio, clip_percentiles)
	    ratio = ratio[(ratio >= lo) & (ratio <= hi)]
	    return np.median(ratio)
	
	def print_params(params):
        amplitude, decay, angle, constant = params
        print(f"amplitude  = {amplitude}")
        print(f"decay      = {decay}")
        print(f"angle      = {angle} rad {angle * 180.0 / np.pi} deg")
        print(f"constant   = {constant}")
    
    def fit_params(self, img, mask):

