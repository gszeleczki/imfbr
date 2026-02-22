import numpy as np
from scipy.optimize import least_squares

# TODO: Add separate linear parameter

class exponential_model:
    def __init__(self, img, mask, settings):
        self.initial_amplitude_clip_percentiles = settings["e_inital_amplitude_clip_percentile_min"], settings["e_inital_amplitude_clip_percentile_max"]
        self.inital_significant_gradient_min_percentile = settings["e_inital_significant_gradient_min_percentile"]
        self.loss_function = settings["e_loss_function"]
        self.method = settings["e_method"]
        self.create_coordinates(img)
        print("Estimating initial values...")
        self.params = self.estimate_initial_values(img, mask)
        self.print_params()
    
    # Calculates the length of the projected gradient vector for each pixel.
    def directional_gradient(self, img, angle, mask):
        dIy, dIx = np.gradient(img)
        grad_dir = np.cos(angle) * dIx + np.sin(angle) * dIy
        return grad_dir[mask]
    
    def create_coordinates(self, img):
        ny, nx = img.shape
        y, x = np.mgrid[0:ny, 0:nx]
    
        self.x = x / (nx - 1)
        self.y = y / (ny - 1)
    
    def calc_diagonal(self, img):
        diagonal_x, diagonal_y = img.shape
        return (diagonal_x ** 2.0 + diagonal_y ** 2.0) ** 0.5
    
    def description(self):
        return "Exponential model in a form of [amplitude * exp((cos(direction) * x + sin(direction) * y) * decay) + constant]."
    
    # Calculates the distance of points from a line which goes through the origin.
    # Returns a 2D array.
    def project_coordinates(self, shape, angle):
        height, width = shape
        y, x = np.indices((height, width))
        projected_coordinates = np.cos(angle) * x + np.sin(angle) * y
        return projected_coordinates
    
    def estimate_decay_and_offset(self, projected_vals, gradient, img_flattened):
        valid_values_mask = np.isfinite(projected_vals) & np.isfinite(gradient) & np.isfinite(img_flattened)
        valid_gradient = gradient[valid_values_mask]
        valid_values = img_flattened[valid_values_mask]
    
        valid_gradient_absolute = np.abs(valid_gradient)
        valid_significant_gradient = valid_gradient_absolute > np.percentile(valid_gradient_absolute, self.inital_significant_gradient_min_percentile)
    
        valid_gradient = valid_gradient[valid_significant_gradient]
        valid_values = valid_values[valid_significant_gradient]
    
        tmp = np.vstack([valid_values, np.ones_like(valid_values)]).T
        decay, intercept = np.linalg.lstsq(tmp, valid_gradient, rcond=None)[0]
    
        offset = -intercept / decay
    
        return decay, offset
    
    def estimate_initial_values(self, img, mask):
        direction = self.estimate_direction(img, mask)
    
        projected_vals = self.project_coordinates(img.shape, direction)[mask]
        img_vals = img[mask]
    
        gradient = self.directional_gradient(img, direction, mask)
        decay, offset = self.estimate_decay_and_offset(projected_vals, gradient, img_vals)
        
        return self.estimate_amplitude(img, mask, direction, decay, offset, self.initial_amplitude_clip_percentiles), decay * self.calc_diagonal(img), direction, offset

    def generate_background(self, params = None):
        local_params = self.params if (params is None) else params 
        coefficient, decay, direction, offset = local_params
        proj = np.cos(direction) * self.x + np.sin(direction) * self.y
        return coefficient * np.exp(decay * proj) + offset

    def estimate_direction(self, img, mask):
        y, x = np.nonzero(mask)
        z = img[y, x]

        X = np.column_stack([
            x,
            y,
            np.ones_like(x)
        ])

        (a, b, c), *_ = np.linalg.lstsq(X, z, rcond=None)

        return np.arctan2(b, a)
        
    def residuals(self, params, img, mask):
        background = self.generate_background(params)
        return (img - background)[mask].ravel()
    
    def estimate_amplitude(self, img, mask, angle, decay, offset, clip_percentiles):
        height, width = img.shape
        y, x = np.mgrid[:height, :width]
    
        projected_coordinates = np.cos(angle) * x + np.sin(angle) * y
        exponentials = np.exp(decay * projected_coordinates)
    
        masked_img = img[mask]
        masked_exponentials = exponentials[mask]
    
        valid_mask = np.isfinite(masked_exponentials) & (masked_exponentials > 0)
        masked_img = masked_img[valid_mask]
        masked_exponentials = masked_exponentials[valid_mask]
    
        ratio = (masked_img - offset) / masked_exponentials
        ratio = ratio[np.isfinite(ratio)]
        
        lo, hi = np.percentile(ratio, self.initial_amplitude_clip_percentiles)
        ratio = ratio[(ratio >= lo) & (ratio <= hi)]
        return np.median(ratio)
    
    def set_params(self, params):
        self.params = params
        
    def print_params(self):
        amplitude, decay, angle, offset = self.params
        print(f"amplitude  = {amplitude}")
        print(f"decay      = {decay}")
        print(f"angle      = {angle} rad {angle * 180.0 / np.pi} deg")
        print(f"offset     = {offset}")
    
    def print_pixelmath_expression(self, img, region_of_interest):
        amplitude, decay, direction, constant = self.params
        cos_angle = np.cos(direction)
        sin_angle = np.sin(direction)
        print(f"$T - {amplitude} * exp({decay / self.calc_diagonal(img)} * ({cos_angle} * x() + {sin_angle} * y())) - {constant} + {np.median(img[region_of_interest])}")
        # $T - 0.003408024138224537 * exp(-0.4752065977993465 * x() + -0.05742004767700772 * y()) - 0.001582727585407239 + 0.004
    
    def fit_params(self, img, region_of_interest):
# TMP42
        result = least_squares(self.residuals, self.params, args = (img, region_of_interest), method = self.method, loss = self.loss_function, verbose = 2, f_scale = np.std(img[region_of_interest]))
        return result.cost, result.x
