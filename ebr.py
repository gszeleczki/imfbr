# IMFBR - Iterative Model Fitting Background Remover

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

DEFAULTS = {
    "dark_absolute_threshold": 0.0001,
    "background_percentile": 20.0,
    "discarded_edge_size": 50,
    "initial_mask_min_structure_size": 6,
    "min_cost_change" = 2e-6,
    "gaussian_smoothing_sigma" = 5.0,
    "inital_A_clip_percentile_min" = 40,
    "inital_A_clip_percentile_max" = 60,
    "min_mask_change_percentile" = 0.1
}

input_path = "integration_WEST_drizzle_B.fit"
dark_absolute_threshold = 0.0001
background_percentile = 20.0
discarded_edge_size = 50
initial_mask_erosion = 6
initial_mask_dilation = 6
min_cost_change = 2e-6
gaussian_smoothing_sigma = 5.0
inital_A_clip_percentiles=(40, 60)
min_mask_change_percentile=0.1

def load_config():
    config_path = "config.ini"
    config = configparser.ConfigParser()
    if path and Path(config_path).exists():
        config.read(path)
        return config["settings"]
    print("Config file [" + config_path + "] not found.")
    return {}

def load_settings():
    parser = argparse.ArgumentParser(description="IMFBR - Iterative Model Fitting Background Remover")

    parser.add_argument("-i", "--input_path", help="Input file path (mandatory).")
    parser.add_argument("--dark_absolute_threshold", type=float, help="Any pixel value below this will be masked out.")
    parser.add_argument("--discarded_edge_size", type=int, help="The initial mask will be dilated by this many pixels (can be used to mask rough edges)")
    parser.add_argument("--initial_mask_min_structure_size", type=int, help="Structures smaller than this many pixels in the initial mask will be removed.")
    parser.add_argument("--min_cost_change", type=float, help="The algorithm terminates if the cost decreases by less than then this amount.")
    parser.add_argument("--min_mask_change_percentile", type=float, help="The algorithm terminates if the mask changes by less than this much percentage.")
    
    parser.add_argument("--inital_amplitude_clip_percentile_min", type=float, help="[exponential model] Min clipping percentile to initially estimate A.")
    parser.add_argument("--inital_amplitude_clip_percentile_max", type=float, help="[exponential model] Max clipping percentile to initially estimate A.")
    parser.add_argument("--inital_significant_gradient_min_percentile", type=float, help="[exponential model] When calculating the gradient, values below this will be not used.")
    parser.add_argument("--gaussian_smoothing_sigma", type=float, help="[exponential model] Smoothing before calculating the gradient for the initial parameter estimation in sigmas.")

    args = parser.parse_args()

    config = load_config(args.config)

    final = {}

    for key, default in DEFAULTS.items():
        cli_value = getattr(args, key)
        cfg_value = config.get(key)

        if cli_value is not None:
            final[key] = cli_value
        elif cfg_value is not None:
            final[key] = type(default)(cfg_value)
        else:
            final[key] = default
            print(f"[INFO] Using default for {key}: {default}")

    print("Final parameters:")
    for k, v in final.items():
        print(f"  {k} = {v}")
    
    return final

def load_fits_image(filename):
    with fits.open(filename) as hdul:
        data = hdul[0].data

    # Convert to float for later math
    return np.asarray(data, dtype=np.float64)

def create_absolute_dark_mask(img, threshold):
    return img > threshold

def create_coordinates(img):
    ny, nx = img.shape
    y, x = np.mgrid[0:ny, 0:nx]

    # Normalize to [0, 1]
    x = x / (nx - 1)
    y = y / (ny - 1)

    return x, y

def background_model(params, x, y):
    A, compression, angle, C = params
    proj = np.cos(angle) * x + np.sin(angle) * y
    return A * np.exp(compression * proj) + C

def residuals(params, x, y, img, mask):
    model = background_model(params, x, y)
    return (img - model)[mask].ravel()

def dark_mask_from_model(
    img,
    model_params,
    model_func,
    x,
    y,
    percentile,
    base_mask
):
    model = model_func(model_params, x, y)

    img_corr = np.full_like(img, np.nan)
    img_corr[base_mask] = img[base_mask] - model[base_mask]

    values = img_corr[base_mask]
    threshold = np.percentile(values, percentile)

    # Build dark mask
    mask_dark = (img_corr <= threshold) & base_mask

    return mask_dark
    
def postprocess_initial_mask(dark_mask, edge_mask, erosion_pixels, dilation_pixels):
    tmp = binary_erosion(dark_mask, iterations=erosion_pixels)
    tmp = binary_dilation(tmp, iterations=dilation_pixels)
    return tmp & edge_mask

def stretch_for_display(img):
    scale = 1.0 / np.std(img)
    scale = np.clip(scale, 3, 20)
    return np.arcsinh(scale * img) / np.arcsinh(scale)

def estimate_angle(img, mask):
    y, x = np.nonzero(mask)
    z = img[y, x]

    X = np.column_stack([
        x,
        y,
        np.ones_like(x)
    ])

    (a, b, c), *_ = np.linalg.lstsq(X, z, rcond=None)

    return np.arctan2(b, a)

def project_coordinates(shape, angle):
    h, w = shape
    y, x = np.indices((h, w))
    s = np.cos(angle) * x + np.sin(angle) * y
    return s

def extract_profile(img, mask, s):
    return s[mask], img[mask]

def smooth_profile(s, values, sigma):
    v_smooth = gaussian_filter1d(values, sigma=sigma)
#    plt.plot(s, v_smooth, '-', lw=2, label="smoothed")
#    plt.plot(s, values, '.', alpha=0.2, label="raw")
#    plt.legend()
#    plt.show()

    return v_smooth

def smooth_image_masked(img, mask, sigma):
    img_filled = img.copy()
    img_filled[~mask] = 0.0

    weight = mask.astype(float)

    img_smooth = gaussian_filter(img_filled, sigma=sigma, mode="reflect")
    weight_smooth = gaussian_filter(weight, sigma=sigma, mode="reflect")

    # Avoid division by zero
    return np.where(weight_smooth > 0,
                    img_smooth / weight_smooth,
                    np.nan)


def derivative(img):
    dy, dx = np.gradient(img)
    return dy, dx

def directional_gradient_2d(img_smooth, angle, mask):
    dIy, dIx = np.gradient(img_smooth)
    grad_dir = np.cos(angle) * dIx + np.sin(angle) * dIy
#    show_image(grad_dir, "grad_dir")
    return grad_dir[mask]

def estimate_compression_and_C(v, dy, dx):
    plt.scatter(v, dy, s=1)
    plt.xlabel("v")
    plt.ylabel("dy")
    plt.show()
    
    plt.scatter(v, dx, s=1)
    plt.xlabel("v")
    plt.ylabel("dx")
    plt.show()
    
    X = np.column_stack([v, np.ones_like(v)])
    (m, b), *_ = np.linalg.lstsq(X, dv, rcond=None)

    compression = m
    C = -b / m if m != 0 else np.median(v)

    return compression, C

def estimate_compression_and_C_new(s, gradient, img_flattened,
                               clip_percentile = 5.0):

    good = np.isfinite(s) & np.isfinite(gradient) & np.isfinite(img_flattened)

    g = gradient[good]
    I = img_flattened[good]

    g_abs = np.abs(g)
    thresh = np.percentile(g_abs, clip_percentile)
    sel = g_abs > thresh

    g = g[sel]
    I = I[sel]

    A = np.vstack([I, np.ones_like(I)]).T
    compression, intercept = np.linalg.lstsq(A, g, rcond=None)[0]

    C = -intercept / compression

    return compression, C

def estimate_A(img, mask, angle, compression, C, clip_percentiles):
    h, w = img.shape
    y, x = np.mgrid[:h, :w]

    proj = np.cos(angle) * x + np.sin(angle) * y
    E = np.exp(compression * proj)

    vals = img[mask]
    E_vals = E[mask]

    valid = np.isfinite(E_vals) & (E_vals > 0)
    vals = vals[valid]
    E_vals = E_vals[valid]

    ratio = (vals - C) / E_vals
    ratio = ratio[np.isfinite(ratio)]
    
    lo, hi = np.percentile(ratio, clip_percentiles)
    ratio = ratio[(ratio >= lo) & (ratio <= hi)]
    return np.median(ratio)

def print_params(params):
    A, compression, angle, C = params
    print(f"A  = {A}")
    print(f"compression = {compression}")
    print(f"angle = {angle} rad {angle * 180.0 / np.pi} deg")
    print(f"C  = {C}")

def calc_diagonal(img):
    diagonal_x, diagonal_y = img.shape
    return (diagonal_x ** 2.0 + diagonal_y ** 2.0) ** 0.5

def estimate_initial_values(img, mask, clip_percentiles):
    angle = estimate_angle(img, mask)

    s = project_coordinates(img.shape, angle)
    s_vals, img_vals = extract_profile(img, mask, s)

#    img_smooth = smooth_profile(s_vals, img_vals, gaussian_smoothing_sigma, mask)
    img_smooth = img#smooth_image_masked(img, mask, gaussian_smoothing_sigma)
#    show_image(img_smooth, "img_smooth")
    gradient = directional_gradient_2d(img_smooth, angle, mask)
    diagonal = calc_diagonal(img)
    compression, C = estimate_compression_and_C_new(s_vals, gradient, img_vals)
    
    return estimate_A(img, mask, angle, compression, C, clip_percentiles), compression * diagonal, angle, C

def show_image(img, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray", origin="lower")
    plt.colorbar(label="Intensity")
    plt.title(title)
    plt.tight_layout()
    plt.show()

settings = load_settings()
print("Loading image: [" + settings["input_path"] + "]")
img = load_fits_image(settings["input_path"])
print("Image loaded.")

print("Dimensions: " + str(img.shape))
print("Sample type: " + str(img.dtype))

print("")
print("Creating absolute dark mask...")
absolute_dark_mask = create_absolute_dark_mask(img, settings["dark_absolute_threshold"])

total_pixels = absolute_dark_mask.size
total_pixels_mp = int(total_pixels / 1000000)
rejected_pixels = (~absolute_dark_mask).sum()
rejected_pixels_mp = int(rejected_pixels / 1000000)
kept_pixels = absolute_dark_mask.sum()
kept_pixels_mp = int(kept_pixels / 1000000)
rejected_pct = 100.0 * rejected_pixels / total_pixels
kept_pct = 100.0 * kept_pixels / total_pixels

print(f"Total pixels: {total_pixels_mp}MP")
print(f"Kept pixels: {kept_pixels_mp}MP ({kept_pct:.2f}%)")
print(f"Rejected pixels: {rejected_pixels_mp}MP ({rejected_pct:.2f}%)")

grown_absolute_dark_mask = binary_erosion(absolute_dark_mask, iterations=discarded_edge_size)

x, y = create_coordinates(img)

initial_mask = dark_mask_from_model(
    img=img,
    model_params=(0.0, 1.0, 0.0, 0.0),
    model_func=background_model,
    x=x,
    y=y,
    percentile=background_percentile,
    base_mask=grown_absolute_dark_mask
)

show_image(initial_mask, "initial mask")

initial_mask = postprocess_initial_mask(initial_mask, grown_absolute_dark_mask, initial_mask_erosion, initial_mask_dilation)

show_image(initial_mask, "initial mask2")

print("Estimating initial values...")
params = estimate_initial_values(img, initial_mask, inital_A_clip_percentiles)

print("Initial values:")
print_params(params)

old_region_of_interest = create_absolute_dark_mask(img, 1.0)
last_cost = float("inf")
improved = True

while improved:
    print("")
    print("***************************")
    print("Fitting background model...")

    print("Calculating region of interest...")
    region_of_interest = dark_mask_from_model(
        img=img,
        model_params=params,
        model_func=background_model,
        x=x,
        y=y,
        percentile=background_percentile,
        base_mask=grown_absolute_dark_mask
    )
    
#    show_image(region_of_interest, "region_of_interest")
#    show_image(background_model(params, x, y), "background_model")
    
    print(f"Median in region of interest: {np.median(img[region_of_interest])}")
    
    only_new = region_of_interest & ~old_region_of_interest
    pct_total = only_new.sum() / region_of_interest.sum() * 100
    print(f"Mask change: {pct_total:.2f}% of image")
    if pct_total < min_mask_change_percentile:
        print("Stopping as background mask converged.")
        improved = False
        break

    result = least_squares(
        residuals,
        params,
        args=(x, y, img, region_of_interest),
        method="trf",
#        loss="soft_l1",
        loss="linear",
        verbose=2,
        ftol=min_cost_change,
        f_scale=np.std(img[region_of_interest])
    #    bounds=bounds,
    )
    
#    params = estimate_initial_values(img, region_of_interest, inital_A_clip_percentiles)
#    new_cost = np.sum(np.abs(img[region_of_interest] - background_model(params, x, y)[region_of_interest]))
    new_cost = result.cost
#    print("New cost: " + str(new_cost))
    
    improved = new_cost < last_cost
    if improved:
        last_cost = new_cost
        params = result.x
        print("New parameters:")
        print_params(params)
        old_region_of_interest = region_of_interest

show_image(background_model(params, x, y), "background_model")
print("Final parameters:")
print_params(params)
print("Pixelmath expression:")
A, compression, angle, C = params
#    proj = np.cos(angle) * x + np.sin(angle) * y
#    return A * np.exp(compression * proj) + C
cos_angle = np.cos(angle)
sin_angle = np.sin(angle)
print(f"$T - {A} * exp({compression / calc_diagonal(img)} * ({cos_angle} * x() + {sin_angle} * y())) - {C} + {np.median(img[absolute_dark_mask])}")

# $T - 0.003408024138224537 * exp(-0.4752065977993465 * x() + -0.05742004767700772 * y()) - 0.001582727585407239 + 0.004

plt.imshow(old_region_of_interest, cmap="gray", origin="upper")
plt.axis("off")
plt.show()

background = background_model(result.x, x, y)

plt.figure(figsize=(15,4))

scale = 1.0 / np.std(img)
scale = np.clip(scale, 3, 20)

plt.subplot(1,3,1)
plt.title("Original image")
img_disp = stretch_for_display(img)
plt.imshow(img_disp, origin="upper", cmap="gray")
plt.colorbar()

plt.subplot(1,3,2)
plt.title("Fitted background")
plt.imshow(background, origin="upper", cmap="gray")
plt.colorbar()

plt.subplot(1,3,3)
plt.title("Residual (img - background)")
residual_disp = stretch_for_display(img - background)
plt.imshow(residual_disp, origin="upper", cmap="gray")
plt.colorbar()

plt.tight_layout()
plt.show()

