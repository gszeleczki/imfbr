import numpy as np

class polynomial_model:

    class polynomial_params:
        def __init__(self, order):
            self.order = order
            self.coefficients = np.zeros(self.order + 1)
            self.direction = 0.0

    def __init__(self, img, order, settings):
        self.shape = img.shape
        self.adaptive_order_enabled = settings["p_adaptive_order"]
        if self.adaptive_order_enabled:
            self.adaptive_order = 1
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

    def project_coordinates(self, direction=None):
        if direction is None:
            direction = self.params.direction()
        return self.x * np.cos(direction) + self.y * np.sin(direction)

    def design_matrix(self, projected_coordinates, order = None):
        if order is None:
            order = self.order
        return np.stack([projected_coordinates**k for k in range(order + 1)], axis=-1)

    # Fit plane to estimate direction
    def estimate_direction_from_plane(self, img, mask):
        x = self.x[mask]
        y = self.y[mask]
        z = img[mask]

        A = np.stack([x, y, np.ones_like(x)], axis=1)

        (a, b, _), *_ = np.linalg.lstsq(A, z, rcond=None)

        return np.arctan2(b, a)

    def generate_background(self, params = None, design_matrix = None):
        local_params = self.params if (params is None) else params

        if design_matrix is None:
            projected_coordinates = self.project_coordinates(local_params.direction)
            design_matrix = self.design_matrix(projected_coordinates)

        return np.tensordot(design_matrix, local_params.coefficients, axes=([-1], [0]))

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
        return new_params

    def least_squares_with_mad(self, img, direction, order, absolute_dark_mask, region_of_interest):
        design_matrix = self.design_matrix(self.project_coordinates(direction), order)
        params, residuals, *_ = self.simple_least_squares(img, direction, order, region_of_interest, design_matrix)

        background = self.generate_background(params, design_matrix)
        mad = np.abs(img - background)[absolute_dark_mask].mean()

        return self.copy_params(params, self.order), residuals, mad

    def adaptive_least_squares(self, img, direction, region_of_interest, absolute_dark_mask):
        # First try with the current adaptive order
        lower_order_params, lower_order_residuals, lower_order_mad = self.least_squares_with_mad(img, direction, self.adaptive_order, absolute_dark_mask, region_of_interest)
        higher_order_params, higher_order_residuals, higher_order_mad = self.least_squares_with_mad(img, direction, self.adaptive_order + 1, absolute_dark_mask, region_of_interest)

        if higher_order_mad < lower_order_mad:
            self.adaptive_order += 1
            print(f"Incrementing polynomial order to {self.adaptive_order}.")

        return (lower_order_params, lower_order_residuals) if higher_order_mad > lower_order_mad else (higher_order_params, higher_order_residuals)

    def simple_least_squares(self, img, direction, order, region_of_interest, design_matrix = None):
        if design_matrix is None:
            design_matrix = self.design_matrix(self.project_coordinates(direction), order)

        params = self.polynomial_params(order)
        params.direction = direction
        params.coefficients, residuals, *_ = np.linalg.lstsq(design_matrix[region_of_interest], img[region_of_interest], rcond = None)
        return params, residuals

    def fit_params(self, img, region_of_interest, absolute_dark_mask):
        direction = self.estimate_direction_from_plane(img, region_of_interest)
        new_params, residuals = self.adaptive_least_squares(img, direction, region_of_interest, absolute_dark_mask) if self.adaptive_order_enabled and self.order != self.adaptive_order else self.simple_least_squares(img, direction, self.order, region_of_interest)
        return np.sqrt(residuals[0] / region_of_interest.sum()), new_params
