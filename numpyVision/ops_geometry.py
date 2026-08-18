import numpy as np

def crop(img_arr, x1, y1, x2, y2):
    return img_arr[y1:y2, x1:x2]

def center_crop(img_arr, crop_width, crop_height):
    H, W = img_arr.shape[:2]
    
    start_y = (H - crop_height) // 2
    start_x = (W - crop_width) // 2
    
    return img_arr[start_y : start_y + crop_height, start_x : start_x + crop_width]

def flip(img_arr, direction="horizontal"):
    if direction == "horizontal":
        return img_arr[:, ::-1]
    elif direction == "vertical":
        return img_arr[::-1, :]
    else:
        print("Invalid direction! Use 'horizontal' or 'vertical'.")
        return img_arr

def rotate_90(img_arr):
    return np.rot90(img_arr, k=-1)

def add_border(img_arr, thickness, color=(0, 0, 0)):
    H, W, C = img_arr.shape[:3]
    
    new_canvas = np.full((H + 2 * thickness, W + 2 * thickness, C), color, dtype=np.uint8)
    
    new_canvas[thickness : thickness + H, thickness : thickness + W] = img_arr
    
    return new_canvas

def mirror_effect(img_arr):
    mirrored = img_arr.copy()
    
    _, W = img_arr.shape[:2]
    mid = W // 2
    
    flipped_left = img_arr[:, :mid][:, ::-1]
    mirrored[:, W - mid : W] = flipped_left
    
    return mirrored