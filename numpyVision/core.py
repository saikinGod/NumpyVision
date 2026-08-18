import matplotlib.pyplot as plt
import numpy as np
from . import ops_color
from . import ops_geometry
from . import ops_filters

class CoreNpVision:
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

    def _handle_inplace(self, new_arr, inplace):
        if inplace:
            self.img_arr = new_arr
            return self
        else:
            return CoreNpVision(new_arr)

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
        
    def adjust_brightness(self, value, inplace=True):
        res = ops_color.adjust_brightness(self.img_arr, value)
        return self._handle_inplace(res, inplace)

    def to_grayscale(self, inplace=True):
        res = ops_color.to_grayscale(self.img_arr)
        return self._handle_inplace(res, inplace)

    def apply_sepia(self, inplace=True):
        res = ops_color.apply_sepia(self.img_arr)
        return self._handle_inplace(res, inplace)
    
    def invert_colors(self, inplace=True):
        res = ops_color.invert_colors(self.img_arr)
        return self._handle_inplace(res, inplace)
    
    def adjust_contrast(self, factor, inplace=True):
        res = ops_color.adjust_contrast(self.img_arr, factor)
        return self._handle_inplace(res, inplace)
    
    def apply_red_tint(self, inplace=True):
        res = ops_color.apply_red_tint(self.img_arr)
        return self._handle_inplace(res, inplace)
    
    def apply_green_tint(self, inplace=True):
        res = ops_color.apply_green_tint(self.img_arr)
        return self._handle_inplace(res, inplace)
    
    def apply_blue_tint(self, inplace=True):
        res = ops_color.apply_blue_tint(self.img_arr)
        return self._handle_inplace(res, inplace)
    
    def solarize(self, threshold=128, inplace=True):
        res = ops_color.solarize(self.img_arr, threshold)
        return self._handle_inplace(res, inplace)
    
    def color_balance(self, r_add=0, g_add=0, b_add=0, inplace=True):
        res = ops_color.color_balance(self.img_arr, r_add, g_add, b_add)
        return self._handle_inplace(res, inplace)

    def crop(self, x1, y1, x2, y2, inplace=True):
        res = ops_geometry.crop(self.img_arr, x1, y1, x2, y2)
        return self._handle_inplace(res, inplace)

    def center_crop(self, crop_width, crop_height, inplace=True):
        res = ops_geometry.crop(self.img_arr, crop_width, crop_height)
        return self._handle_inplace(res, inplace)

    def flip(self, direction="horizontal", inplace=True):
        res = ops_geometry.flip(self.img_arr, direction)
        return self._handle_inplace(res, inplace)

    def rotate_90(self, inplace=True):
        res = ops_geometry.rotate_90(self.img_arr)
        return self._handle_inplace(res, inplace)

    def add_border(self, thickness, color=(0, 0, 0), inplace=True):
        res = ops_geometry.add_border(self.img_arr, thickness, color)
        return self._handle_inplace(res, inplace)

    def mirror_effect(self, inplace=True):
        res = ops_geometry.mirror_effect(self.img_arr)
        return self._handle_inplace(res, inplace)

    def blur(self,intensity, inplace=True):
        res = ops_filters.blur(self.img_arr,intensity)
        return self._handle_inplace(res, inplace)

    def edge_detect(self, inplace=True):
        res = ops_filters.edge_detect(self.img_arr)
        return self._handle_inplace(res, inplace)

    def sharpen(self, inplace=True):
        res = ops_filters.sharpen(self.img_arr)
        return self._handle_inplace(res, inplace)

    def emboss(self, inplace=True):
        res = ops_filters.emboss(self.img_arr)
        return self._handle_inplace(res, inplace)