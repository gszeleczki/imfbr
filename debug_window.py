import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class imfbr_debug_window:
    def __init__(self, original_image, absolute_dark_mask):
        plt.ion()  # non-blocking UI

        self.shape = original_image.shape
        black = np.zeros(self.shape)

        self.absolute_dark_mask = absolute_dark_mask

        self.fig, axes = plt.subplots(2, 2)
        self.fig.canvas.manager.set_window_title("IMFBR debug window")

        self.axes = axes.flatten()
        self.images = []
        self.absolute_mask = True
        self.stopped = False

        self.default_histogram_params = self.histogram_params(original_image)
        img = self.axes[0].imshow(self.stf_stretch(original_image, self.default_histogram_params), cmap="gray", vmin=0, vmax=1)
        self.axes[0].set_axis_off()
        self.images.append(img)

        # Remaining 3 images (black initially)
        for i in range(1, 4):
            img = self.axes[i].imshow(black, cmap="gray", vmin=0, vmax=1)
            self.axes[i].set_axis_off()
            self.images.append(img)

        self.axes[0].set_title("Original image")
        self.axes[1].set_title("Corrected image")
        self.axes[2].set_title("Background")
        self.axes[3].set_title("Initial mask")

        self.fig.tight_layout()

        self.fig.subplots_adjust(bottom = 0.15)
        button_ax = self.fig.add_axes([0.4, 0.02, 0.2, 0.07])
        self.button = Button(button_ax, "Stop after\niteration finished")
        self.button.on_clicked(self._button_clicked)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def histogram_params(self, image, shadows_clip_sigma = 2.8):
        median = np.median(image[self.absolute_dark_mask])
        mad = np.median(np.abs(image[self.absolute_dark_mask] - median))
        sigma = 1.4826 * mad

        shadows = median - shadows_clip_sigma * sigma
        shadows = max(shadows, image[self.absolute_dark_mask].min())

        highlights = image[self.absolute_dark_mask].max()
        return shadows, highlights

    def stf_stretch(self, image, histogram_params = None, shadows_clip_sigma = 2.8, target_background = 0.25):
        shadows, highlights = histogram_params if not (histogram_params is None) else self.histogram_params(image, shadows_clip_sigma)
        img = image.astype(np.float32)

        # normalize
        x = (img - shadows) / (highlights - shadows)
        x = np.clip(x, 0, 1)
        median = np.median(x[self.absolute_dark_mask])

        # midtones parameter
        m = (median * (1.0 - target_background)) / (median + target_background - 2.0 * median * target_background)

        # midtones transfer function
        stretched = ((m - 1) * x) / ((2 * m - 1) * x - m)
        stretched = np.clip(stretched, 0, 1)

        return stretched

    def _button_clicked(self, event):
        self.stopped = True

    def update_corrected_image(self, image):
        self.images[1].set_data(self.stf_stretch(image))
        self._refresh()

    def update_background_image(self, image):
        self.images[2].set_data(self.stf_stretch(image, self.default_histogram_params))
        self._refresh()

    def update_region_of_interest(self, image):
        if self.absolute_mask:
            self.axes[3].set_title("Current region of interest")
            self.absolute_mask = False
        self.images[3].set_data(image)
        self._refresh()

    def update_absolute_mask_image(self, image):
        if not self.absolute_mask:
            self.axes[3].set_title("Initial mask")
            self.absolute_mask = True
        self.update_region_of_interest(image)

    def _refresh(self):
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def on_finished(self):
        print("Close the debug window to finish the script.")
        self.fig.canvas.manager.set_window_title("Close this window to finish the script.")
        self.button.label.set_text("Stopped.")
        self._refresh()
        plt.ioff()        # turn interactive OFF
        plt.show()        # now blocks until user closes window
