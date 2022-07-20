'''
Various utils.
'''

from typing import List, Dict, Any, Tuple
import math

import cv2
from matplotlib import pyplot as plt
import numpy as np

def elapsed_ms(start_time: float) -> int:
    '''Return ms elapsed since passed start time (use time.time()).'''
    return round((time.time() - start_time)*1000)

def display_images(images: List[Dict[str, Any]]) -> None:
    '''Display debug images in a popup.'''
    plot_cols = None
    plot_rows = None
    if len(images) <= 3:
        plot_rows = 1
    elif len(images) <= 8:
        plot_rows = 2
    else:
        plot_rows = 3
    plot_cols = math.ceil(len(images)/plot_rows)
    for i, display_img_info in enumerate(images):
        plt.subplot(plot_rows, plot_cols, i + 1)
        plt.imshow(display_img_info['img'], cmap='gray')
        plt.title(display_img_info['title'])
        plt.xticks([]), plt.yticks([])

    plt.show()

def img_threshold_by_range(img: np.array, min: List, max: List, reverse: bool = True) -> np.array:
    '''Given input image and list of mins and max pixel values, return thresholded image.
    
    Args:
        img: input cv2.Mat
        min: list of min pixel values for all img channes
        max: list of max pixel values for all img channes

    Returns: thresholded binary cv2.Mat

    Note: try just using cv2.inRange() as done in the answer code here: https://stackoverflow.com/a/52048325/17591909
    '''

    min = min.copy()
    max = max.copy()

    channel_count = 1 if len(img.shape) == 2 else img.shape[2]
    assert len(min) == len(max) and len(min) == channel_count, \
        f'mismatched channel counts: min = {len(min)}, max = {len(max)}, img = {channel_count}'
    
    # min, max are RGB but cv2 stores images as BGR
    if reverse:
        min.reverse()
        max.reverse()

    img_sample_val = img[0][0] if channel_count == 1 else img[0][0][0]
    mask_type = type(img_sample_val)
    mask_shape = (img.shape[0], img.shape[1])
    result = np.full(mask_shape, True)
    for i in range(channel_count):
        img_mask = img if channel_count == 1 else img[:,:,i]
        min_mask = np.full(mask_shape, min[i], mask_type)
        max_mask = np.full(mask_shape, max[i], mask_type)
        mask_result = cv2.inRange(img_mask, min_mask, max_mask)
        result = np.logical_and(result, mask_result)

    return np.uint8(result)*255

def get_mask_white_pixels(mask: np.array) -> np.array:
    '''Returns list of white pixels in a binary image as a 2xN np array.
    
    Args:
        mask: np.array representing binary image with type uint8

    Return: List of white pixels in mask with format: [[x0, y0], [x1, y1], ...]

    Note: returns np.array not List[Point] b/c want to use np utils downstream
    '''
    assert mask.dtype == np.uint8, f'mask must be of type np.uint8, found {mask.dtype}'

    raw_mask_pixels = np.where(mask == [255])
    return np.column_stack((raw_mask_pixels[1], raw_mask_pixels[0]))  # reverse order to get [x, y] points

def get_pixel_extremes(pixels: np.array) -> Tuple[int, int, int, int]:
    '''Given a list of pixels returns extreme values.
    
    Args:
        pixels: 2xN np.array with pixels, format: [[x0, y0], [x1, y1], ...]

    Returns:
        1. xmin
        2. xmax
        3. ymin
        4. ymax
    '''

    assert pixels.shape[1] == 2, 'pixels must be of format: [[x0, y0], [x1. y1], ...]'

    xmin_idx, ymin_idx = np.argmin(pixels, axis=0)
    xmax_idx, ymax_idx = np.argmax(pixels, axis=0)

    xmin = pixels[xmin_idx][0]
    xmax = pixels[xmax_idx][0]
    ymin = pixels[ymin_idx][1]
    ymax = pixels[ymax_idx][1]

    return (xmin, xmax, ymin, ymax)