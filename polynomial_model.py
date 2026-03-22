import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

class polynomial_model:

    class polynomial_params:
        def __init__(self, order):
            self.order = order
            self.coefficients = np.zeros(self.order + 1)
            self.direction = 0.0
            self.background_cache = None

    class fit_params_environment:
        def __init__(self, model, img, region_of_interest, absolute_dark_mask, order):
            self.order = order
            self.model = model
            self.img = img
            self.region_of_interest = region_of_interest
            self.absolute_dark_mask = absolute_dark_mask

            self.masked_img = self.img[self.region_of_interest]
            self.direction = self.model.estimate_direction_from_plane(self.masked_img, self.region_of_interest)
            self.projected_coordinates = self.model.project_coordinates(self.direction)

    def __init__(self, img, order, settings):
        self.shape = img.shape
        self.dtype = img.dtype
        self.background_buffer = np.zeros_like(img)
        self.mad_buffer = np.zeros_like(img)
        self.adaptive_order_enabled = settings["p_adaptive_order"]
        if self.adaptive_order_enabled:
            self.adaptive_order = 1
            self.higher_order_background_buffer = np.zeros_like(img)
            self.higher_order_mad_buffer = np.zeros_like(img)
        self.order = order
        self.params = self.polynomial_params(order)

        self.x, self.y = self.create_coords()

    def description(self):
        return "Constrained polynomial model. It assumes the background is monotone, and it can be modeled as a 1D polynomial along a direction."

    def create_coords(self):
        h, w = self.shape
        y, x = np.indices((h, w))

        x = (x / (w - 1)) * 2 - 1
        y = (y / (h - 1)) * 2 - 1

        return x, y

    def project_coordinates(self, direction):
        return self.x * np.cos(direction) + self.y * np.sin(direction)

    def estimate_direction_from_plane(self, masked_img, mask):
        x = self.x[mask]
        y = self.y[mask]
        z = masked_img

        n = x.size

        Sx  = x.sum()
        Sy  = y.sum()
        Sz  = z.sum()

        Sxx = (x * x).sum()
        Syy = (y * y).sum()
        Sxy = (x * y).sum()

        Sxz = (x * z).sum()
        Syz = (y * z).sum()

        M = np.array([
            [Sxx, Sxy, Sx],
            [Sxy, Syy, Sy],
            [Sx,  Sy,  n ]
        ])

        rhs = np.array([Sxz, Syz, Sz])

        a, b, c = np.linalg.solve(M, rhs)

        return np.arctan2(b, a)

    def generate_background(self, params = None, environment = None, buffer = None):
        local_params = self.params if (params is None) else params
        if local_params.background_cache is None:
            projected_coordinates = self.project_coordinates(local_params.direction) if environment is None else environment.projected_coordinates
            if buffer is None:
                result = np.zeros_like(projected_coordinates)
            else:
                result = buffer
                result.fill(0.0)
            for coeff in reversed(local_params.coefficients):
               result *= projected_coordinates
               result += coeff
            local_params.background_cache = result
        return local_params.background_cache

    def print_params(self, params = None):
        if params is None:
            params = self.params
        print(f"direction = {params.direction:.6f} rad ({np.degrees(params.direction):.2f} deg)")
        for k, p in enumerate(params.coefficients):
            print(f"coefficient of order {k} = {p:.6g}")

    def pixelmath_expression(self):
        projected_expr = f"(((2 * x() / (w() - 1)) - 1.0) * {np.cos(self.params.direction)} + ((2 * y() / (h() - 1)) - 1.0) * {np.sin(self.params.direction)})"
        terms = []
        for order, coefficient in enumerate(self.params.coefficients):
            if coefficient == 0.0:
                continue

            if order == 0:
                term = str(coefficient)
            elif order == 1:
                term = f"{coefficient}*{projected_expr}"
            else:
                term = f"{coefficient}*{projected_expr}^{order}"

            terms.append(term)

        if not terms:
            terms = ["0"]

        expr = " + ".join(terms).replace("+ -", "- ")
        return expr

    def set_params(self, params):
        self.params = params

    def copy_params(self, params, order):
        new_params = self.polynomial_params(order)
        new_params.direction = params.direction
        for order_to_copy in range(0, min(order + 1, len(params.coefficients))):
            new_params.coefficients[order_to_copy] = params.coefficients[order_to_copy]
        new_params.background_cache = params.background_cache
        return new_params

    def least_squares_with_mad(self, environment, higher_order):
        params, residuals, *_ = self.simple_least_squares(environment, higher_order)

        background = self.generate_background(params, environment, self.higher_order_background_buffer if higher_order else self.background_buffer)
        mad_buffer = self.higher_order_mad_buffer if higher_order else self.mad_buffer

        np.subtract(environment.img, background, out=mad_buffer)
        np.abs(mad_buffer, out=mad_buffer)

        mad = mad_buffer[environment.absolute_dark_mask].sum() / environment.absolute_dark_mask.sum()

        return self.copy_params(params, self.order), residuals, mad

    def adaptive_least_squares(self, environment):
        # First try with the current adaptive order
        with ThreadPoolExecutor() as executor:
            future_lower_order = executor.submit(self.least_squares_with_mad, environment, False)
            future_higher_order = executor.submit(self.least_squares_with_mad, environment, True)

            lower_order_params, lower_order_residuals, lower_order_mad = future_lower_order.result()
            higher_order_params, higher_order_residuals, higher_order_mad = future_higher_order.result()

        if higher_order_mad < lower_order_mad:
            self.adaptive_order += 1
            print(f"Incrementing polynomial order to {self.adaptive_order}.")

        return (lower_order_params, lower_order_residuals) if higher_order_mad > lower_order_mad else (higher_order_params, higher_order_residuals)

    def simple_least_squares(self, environment, higher_order):
        order = (environment.order + 1) if higher_order else environment.order
        params = self.polynomial_params(order)
        params.direction = environment.direction

        x = environment.projected_coordinates[environment.region_of_interest]
        z = environment.img[environment.region_of_interest]

        n = order + 1

        ATA = np.zeros((n, n))
        ATz = np.zeros(n)

        powers = [np.ones_like(x)]
        for _ in range(order):
            powers.append(powers[-1] * x)

        for i in range(n):
            xi = powers[i]

            ATz[i] = (xi * z).sum()

            for j in range(i, n):
                val = (xi * powers[j]).sum()
                ATA[i, j] = val
                ATA[j, i] = val

        params.coefficients = np.linalg.solve(ATA, ATz)
        result = np.zeros_like(x)

        for coeff in reversed(params.coefficients):
            result *= x
            result += coeff

        return params, [((result - z) ** 2).sum()]

    def fit_params(self, img, region_of_interest, absolute_dark_mask):
        environment = self.fit_params_environment(self, img, region_of_interest, absolute_dark_mask, (self.adaptive_order if self.adaptive_order_enabled else self.order))
        new_params, residuals = self.adaptive_least_squares(environment) if (self.adaptive_order_enabled and self.order != self.adaptive_order) else self.simple_least_squares(environment, False)
        return np.sqrt(residuals[0] / region_of_interest.sum()), new_params
