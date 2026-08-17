import numpy as np

def adjust_brightness(img_arr, value):
    temp_arr = img_arr.astype(np.int16)
    clipped_arr = np.clip(temp_arr + value, 0, 255)

    return clipped_arr.astype(np.uint8)


def to_grayscale(img_arr):
    return np.mean(img_arr, axis = 2).astype(np.uint8)

def apply_sepia(img_arr):
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])

    result = np.dot(img_arr, sepia_matrix.T)
    result = np.clip(result, 0, 255)

    return result.astype(np.uint8)

def invert_colors(img_arr):
    inverted = 255 - img_arr
    return inverted.astype(np.uint8)

def adjust_contrast(img_arr, factor):
    temp_arr = img_arr.astype(np.float32)
    temp_arr = (temp_arr - 128) * factor + 128 
    
    clipped = np.clip(temp_arr, 0, 255)
    return clipped.astype(np.uint8)

def apply_red_tint(img_arr):
    tinted = img_arr.copy()
    
    tinted[:, :, 1] = 0
    tinted[:, :, 2] = 0
        
    return tinted
    
def apply_green_tint(img_arr):
    tinted = img_arr.copy()

    tinted[:, :, 0] = 0
    tinted[:, :, 2] = 0

    return tinted

def apply_blue_tint(img_arr):
    tinted = img_arr.copy()

    tinted[:, :, 0] = 0
    tinted[:, :, 1] = 0

    return tinted

def solarize(img_arr, threshold=128):
    solarized = np.where(img_arr > threshold, 255 - img_arr, img_arr)
    return solarized.astype(np.uint8)

def color_balance(img_arr, r_add=0, g_add=0, b_add=0):
    temp_arr = img_arr.astype(np.int16)

    temp_arr[:, :, 0] += r_add
    temp_arr[:, :, 1] += g_add
    temp_arr[:, :, 2] += b_add

    balanced = np.clip(temp_arr, 0, 255)

    return balanced.astype(np.uint8)
