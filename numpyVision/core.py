import matplotlib.pyplot as plt
import numpy as np
from . import ops_color
from . import ops_geometry
from . import ops_filters

class CorenpVision:
    def __init__(self, img_data):
        if isinstance(img_data, str):
            self.img_arr = plt.imread(img_data)
        else:
            self.img_arr = img_data.copy()

        # PNG/Float
        if self.img_arr.dtype == np.float32 or self.img_arr.dtype == np.float64:
            self.img_arr = (self.img_arr * 255).astype(np.uint8)

        # RGBA Alpha Channel
        if len(self.img_arr.shape) == 3 and self.img_arr.shape[-1] == 4:
            self.img_arr = self.img_arr[:, :, :3]

    def save(self, new_path):
        plt.imsave(new_path, self.img_arr)

    def show(self, title="Your Image"):
        plt.figure(figsize=(8, 6))
        plt.title(title, fontsize=14, fontweight='bold', pad=15)
        
        if len(self.img_arr.shape) == 2:
            plt.imshow(self.img_arr, cmap='gray')
        else:
            plt.imshow(self.img_arr)

        plt.axis('off')
        plt.tight_layout()
        plt.show()