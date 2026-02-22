# IMFBR - Iterative Model Fitting Background Remover

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation
from scipy.ndimage import gaussian_filter
import argparse
import configparser
import sys
from pathlib import Path
import re
import exponential_model as em

class imfbr:
    ALL_PARAMETERS = {
        "input_path": "",
        "dark_absolute_threshold": 0.0001,
        "background_percentile": 10.0,
        "discarded_edge_size": 50,
        "mask_min_structure_size": 6,
        "mask_structure_growth": 1,
        "min_cost_change" : 2e-6,
        "model_type" : "e",
        "print_pixelmath_expression" : True,
        
        "e_inital_amplitude_clip_percentile_min" : 40,
        "e_inital_amplitude_clip_percentile_max" : 60,
        "e_inital_significant_gradient_min_percentile" : 5.0,
        "e_loss_function" : "linear",
        "e_method" : "trf"
    }
    
    def load_config(self):
        config_path = Path("config.ini")
        config = configparser.ConfigParser()
        if config_path.exists():
            config.read(config_path)
            return config["settings"]
        print("Config file [" + str(config_path) + "] not found.")
        return {}
    
    def load_settings(self):
        parser = argparse.ArgumentParser(description="IMFBR - Iterative Model Fitting Background Remover")
    
        parser.add_argument("-i", "--input_path", type=str, help="Input file path (mandatory).")
        parser.add_argument("-m", "--model_type", type=str, help="Background model:\n  e: Exponential model in a form of [amplitude * exp((cos(direction) * x + sin(direction) * y) * decay) + constant]\n  p[n]: Polynomial with order n")
        parser.add_argument("-bp", "--background_percentile", type=float, help="The percentile of the area which can be considered as background.")
        parser.add_argument("--dark_absolute_threshold", type=float, help="Any pixel value below this will be masked out.")
        parser.add_argument("--discarded_edge_size", type=int, help="The initial mask will be dilated by this many pixels (can be used to mask rough edges)")
        parser.add_argument("--mask_min_structure_size", type=int, help="Structures smaller than this many pixels in the mask will be removed.")
        parser.add_argument("--mask_structure_growth", type=int, help="The unmasked area growth in pixels.")
        parser.add_argument("--min_cost_change", type=float, help="The algorithm terminates if the cost decreases by less than then this amount.")
        parser.add_argument("--print_pixelmath_expression", type=bool, help="Print the pixelmath expression at the end which correct the background.")
        
        parser.add_argument("--e_inital_amplitude_clip_percentile_min", type=float, help="[exponential model] Min clipping percentile to initially estimate the amplitude.")
        parser.add_argument("--e_inital_amplitude_clip_percentile_max", type=float, help="[exponential model] Max clipping percentile to initially estimate the amplitude.")
        parser.add_argument("--e_inital_significant_gradient_min_percentile", type=float, help="[exponential model] When calculating the gradient, values below this will be not used.")
        parser.add_argument("--e_loss_function", type=str, help="[exponential model] The loss function - see scipy documentation of least squares for possible values.")
        parser.add_argument("--e_method", type=str, help="[exponential model] The fitting method - see scipy documentation of least squares for possible values.")
    
        args = parser.parse_args()
    
        config = self.load_config()
        if len(config) == 0:
            exit()
    
        self.settings = {}
    
        for key, default in self.ALL_PARAMETERS.items():
            cli_value = getattr(args, key)
            cfg_value = config.get(key)
    
            if cli_value is not None:
                self.settings[key] = cli_value
            elif cfg_value is not None:
                self.settings[key] = type(default)(cfg_value)
            else:
                self.settings[key] = default
                print(f"[INFO] Using default for {key}: {default}")
    
        print("Run parameters:")
        for k, v in self.settings.items():
            print(f"  {k} = {v}")
        
        self.model_type = self.settings["model_type"]
        self.input_path = self.settings["input_path"]
        self.dark_absolute_threshold = self.settings["dark_absolute_threshold"]
        self.background_percentile = self.settings["background_percentile"]
        self.discarded_edge_size = self.settings["discarded_edge_size"]
        self.mask_min_structure_size = self.settings["mask_min_structure_size"]
        self.mask_structure_growth = self.settings["mask_structure_growth"]
        self.min_cost_change = self.settings["min_cost_change"]
        self.print_pixelmath_expression = self.settings["print_pixelmath_expression"]
    
    def load_fits_image(self, filename = None):
        if filename is None:
            filename = self.input_path
        with fits.open(filename) as hdul:
            data = hdul[0].data
    
        return np.asarray(data, dtype=np.float64)
    
    def create_absolute_dark_mask(self, img, threshold):
        return img > threshold
    
    def create_model_dark_mask(self, background):
        if background is None:
            background = np.zeros_like(self.img)
    
        img_corr = np.full_like(self.img, np.nan)
        img_corr[self.grown_absolute_dark_mask] = self.img[self.grown_absolute_dark_mask] - background[self.grown_absolute_dark_mask]
    
        values = img_corr[self.grown_absolute_dark_mask]
        threshold = np.percentile(values, self.background_percentile)
    
        intermediate = ((img_corr <= threshold) & self.grown_absolute_dark_mask)
        intermediate = binary_erosion(intermediate, iterations = self.mask_min_structure_size)
        intermediate = binary_dilation(intermediate, iterations = self.mask_min_structure_size + self.mask_structure_growth)
        
        return intermediate & self.grown_absolute_dark_mask
            
    def calc_diagonal(self):
        diagonal_x, diagonal_y = self.img.shape
        return (diagonal_x ** 2.0 + diagonal_y ** 2.0) ** 0.5
    
    def show_image(self, img, title):
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap="gray", origin="lower")
        plt.colorbar(label="Intensity")
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def run(self):
        self.load_settings()
        
        print("")
        print("Loading image: [" + self.input_path + "]")
        self.img = self.load_fits_image()
        print("Image loaded.")
        
        print("Dimensions: " + str(self.img.shape))
        print("Sample type: " + str(self.img.dtype))
        
        print("")
        print("Creating absolute dark mask...")
        self.absolute_dark_mask = self.create_absolute_dark_mask(self.img, self.dark_absolute_threshold)
        
        total_pixels = self.absolute_dark_mask.size
        total_pixels_mp = int(total_pixels / 1000000)
        rejected_pixels = (~self.absolute_dark_mask).sum()
        rejected_pixels_mp = int(rejected_pixels / 1000000)
        kept_pixels = self.absolute_dark_mask.sum()
        kept_pixels_mp = int(kept_pixels / 1000000)
        rejected_pct = 100.0 * rejected_pixels / total_pixels
        kept_pct = 100.0 * kept_pixels / total_pixels
        
        print(f"Total pixels: {total_pixels_mp}MP")
        print(f"Kept pixels: {kept_pixels_mp}MP ({kept_pct:.2f}%)")
        print(f"Rejected pixels: {rejected_pixels_mp}MP ({rejected_pct:.2f}%)")
        
        self.grown_absolute_dark_mask = binary_erosion(self.absolute_dark_mask, iterations = self.discarded_edge_size)
        
        self.model = None
        if self.model_type == "e":
            print("Using exponential model.")
            self.model = em.exponential_model(self.img, self.grown_absolute_dark_mask, self.settings)
        elif (m := re.fullmatch(r"p(\d)", self.model_type)):
            print(f"Using {m.group(1)}-order polynomial model.")
        #    model = polym.polynomial_model(int(m.group(1)))
        else:
            print("Unknown model: " + self.model_type)
            exit()
        
        union_region_of_interest = np.zeros_like(self.img, dtype=bool)
        region_of_interest = None
        last_cost = float("inf")
        improved = True
        
        while improved:
            print("")
            print("***************************")
            print("Fitting background model...")
        
            print("Calculating region of interest...")
            region_of_interest = self.create_model_dark_mask(None if (region_of_interest is None) else self.model.generate_background())
            
            print(f"Median in region of interest: {np.median(self.img[region_of_interest])}")
            
            new_pixels = region_of_interest & ~union_region_of_interest            
            if not new_pixels.any():
                print("Stopping as background mask doesn't have new pixels.")
                improved = False
                break
            else:
                pct_total = new_pixels.sum() / region_of_interest.sum() * 100
                print(f"Mask change: {pct_total:.2f}% of image")
            
            new_cost, new_params = self.model.fit_params(self.img, region_of_interest)
            print(f"New cost: {new_cost} Last cost: {last_cost}")
            
            const_improvement = last_cost - new_cost
            improved = const_improvement > 0.0
            if improved:
                last_cost = new_cost
                self.model.set_params(new_params)
                print("New parameters:")
                self.model.print_params()
                union_region_of_interest |= region_of_interest
            
            if const_improvement < self.min_cost_change:
                print("Stopping as cost improvement is less than the threshold.")
                improved = False
        
        print("Final parameters:")
        self.model.print_params()
        if self.print_pixelmath_expression:
            print("Pixelmath expression:")
            self.model.print_pixelmath_expression(self.img, region_of_interest)

instance = imfbr()
instance.run()
