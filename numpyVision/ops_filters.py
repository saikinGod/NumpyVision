import numpy as np

def _apply_3x3_kernel(img_arr, kernel):
    H, W = img_arr.shape[:2]
    
    padded = np.pad(img_arr, ((1, 1), (1, 1), (0, 0)), mode='edge').astype(np.float32)
    
    result = np.zeros_like(img_arr, dtype=np.float32)
    
    for i in range(3):
        for j in range(3):
            result += padded[i : i+H, j : j+W] * kernel[i, j]
            
    return np.clip(result, 0, 255).astype(np.uint8)

def blur(img_arr, intensity):
    blur_kernel = np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ])
    
    blurred_img = img_arr.copy()

    for _ in range(intensity):
        blurred_img = _apply_3x3_kernel(blurred_img, blur_kernel)
        
    return blurred_img

def edge_detect(img_arr):
    edge_kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])

    return _apply_3x3_kernel(img_arr, edge_kernel)

def sharpen(img_arr):
    sharpen_kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
    return _apply_3x3_kernel(img_arr, sharpen_kernel)

def emboss(img_arr):
    emboss_kernel = np.array([
        [-2, -1,  0],
        [-1,  1,  1],
        [ 0,  1,  2]
    ])
    return _apply_3x3_kernel(img_arr, emboss_kernel)