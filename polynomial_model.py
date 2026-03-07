import numpy as np

class polynomial_model:
    def __init__(self, img, order):
        self.shape = img.shape
        self.order = order
        self.params = np.zeros(self.order + 2)

        self.x, self.y = self.create_coords()

    def direction(self):
        return self.params[-1]

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
            direction = self.direction()
        return self.x * np.cos(direction) + self.y * np.sin(direction)

    def design_matrix(self, projected_coordinates):
        return np.stack([projected_coordinates**k for k in range(self.order + 1)], axis=-1)

    # Fit plane to estimate direction
    def estimate_direction_from_plane(self, img, mask):
        x = self.x[mask]
        y = self.y[mask]
        z = img[mask]

        A = np.stack([x, y, np.ones_like(x)], axis=1)

        (a, b, _), *_ = np.linalg.lstsq(A, z, rcond=None)

        return np.arctan2(b, a)

    def generate_background(self, params=None):
        local_params = self.params if (params is None) else params

        projected_coordinates = self.project_coordinates()
        A = self.design_matrix(projected_coordinates)

        return np.tensordot(A, local_params[:-1], axes=([-1], [0]))

    def print_params(self):
        print(f"direction = {self.direction():.6f} rad ({np.degrees(self.direction()):.2f} deg)")
        for k, p in enumerate(self.params):
            if k != len(self.params) -1:
                print(f"coefficient of order {k} = {p:.6g}")

    def pixelmath_expression(self):
        projected_expr = f"(((2 * x() / (w() - 1)) - 1.0) * {np.cos(self.params[-1])} + ((2 * y() / (h() - 1)) - 1.0) * {np.sin(self.params[-1])})"
        terms = []
        for order, coefficient in enumerate(self.params):
            if order == len(self.params) - 1:
                break
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

    def fit_params(self, img, region_of_interest):
        new_params = np.zeros(self.order + 2)
        new_params[-1] = self.estimate_direction_from_plane(img, region_of_interest)
        design_matrix = self.design_matrix(self.project_coordinates(self.direction()))

        new_params[:-1], *_ = np.linalg.lstsq(design_matrix[region_of_interest], img[region_of_interest], rcond=None)

        diff = img - self.generate_background(new_params)

        return np.mean((diff[region_of_interest])**2), new_params
