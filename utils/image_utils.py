'''Various utils for parsing images.'''

import sys
from os import path
from typing import List, Tuple
import copy
sys.path.append(path.join(path.dirname(__file__), '..')) # upwards relative imports are hacky
sys.path.append(path.join(path.dirname(__file__), '..')) # upwards relative imports are hacky

import cv2
import numpy as np
import skimage
from skimage.morphology import skeletonize

import constants
from plays.play import Play, PlayMask, Point

ROUTE_COLORS = [
    [255,0,0],
    [0,255,0],
    [0,0,255],
    [255,255,0],
    [255,0,255],
    [0,255,255],
    [255,255,255],
]


def img_threshold_by_range(img: np.array, min: List, max: List, reverse: bool = True) -> np.array:
    '''Given input image and list of mins and max pixel values, return thresholded image.
    
    Can threshold grayscale, rgb, hsv images and more.

    Args:
        img: input cv2.Mat
        min: list of min pixel values for all img channels
        max: list of max pixel values for all img channels

    Returns: thresholded uint8 binary cv2.Mat

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

    raw_mask_pixels = np.where(mask > [0])
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


def crop_image(input: np.array, crop_fracs: Tuple[float, float, float, float]) -> Tuple[np.array, np.array]:
    '''Return cropped image.
    
    Args:
        input: input frame to crop
        crop_fracs: [xmin, ymin, xmax, ymax] from top left as fractions

    Return:
        1. cropped image
        2. debug image - uncropped image with cropbox drawn
    '''
    height, width, _ = input.shape
    crop_pxs = [
        int(crop_fracs[0]*width),
        int(crop_fracs[1]*height),
        int(crop_fracs[2]*width),
        int(crop_fracs[3]*height),
    ]
    crop_img = input[crop_pxs[1]:crop_pxs[3], crop_pxs[0]:crop_pxs[2]]

    cropbox_img = input.copy()
    cropbox_corners = np.array([
        [crop_pxs[0], crop_pxs[1]],
        [crop_pxs[2], crop_pxs[1]],
        [crop_pxs[2], crop_pxs[3]],
        [crop_pxs[0], crop_pxs[3]],
        [crop_pxs[0], crop_pxs[1]],
    ]).astype(np.int32)
    cv2.polylines(cropbox_img, [cropbox_corners], False, constants.DEBUG_COLOR, 1)

    return (crop_img, cropbox_img)


def parse_playmask_from_masks(masks: List[np.array], ball_location: Point, playmask_path: str) -> Tuple[PlayMask, List[np.array]]:
    '''Given a list of cleaned masks parse into a combined PlayMask.
    
    Args:
        masks: list of binary masks to parse routes from

    Return:
        1. parsed PlayMask
        2. list of debug images
    '''

    assert len(masks) > 0
    masks = list(masks)
    mask_shape = masks[0].shape

    debug_images = []
    skeletons_image = np.zeros([mask_shape[0], mask_shape[1], 3], np.uint8)

    play_mask = np.zeros(mask_shape, np.uint8)
    contour_idx = -1
    for i, mask in enumerate(masks):
        contours, _ = cv2.findContours(mask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        all_mask_contours_image = np.zeros(mask_shape, np.uint8)
        all_mask_contours_image = cv2.drawContours(all_mask_contours_image, contours, -1, constants.DEBUG_COLOR, -1)
        # debug_images.append({'title': f'all contour {i}', 'img': all_mask_contours_image})
        
        for j, contour in enumerate(contours):
            contour_idx += 1
            color_idx = contour_idx % len(ROUTE_COLORS)
            debug_color = ROUTE_COLORS[color_idx]
            
            contour_image = np.zeros(mask_shape, np.uint8)
            contour_image = cv2.drawContours(contour_image, [contour], 0, (255, 255, 255), -1)
            # debug_images.append({'title': f'contour {contour_idx}', 'img': contour_image.copy()})

            skeletonized = skeletonize(skimage.img_as_float(contour_image.copy())).astype('uint8') * 255
            play_mask = cv2.bitwise_or(play_mask, skeletonized)

            # Create skeletons debug image
            colorized_skeleton = cv2.cvtColor(skeletonized, cv2.COLOR_GRAY2RGB)
            colorized_skeleton[:,:,0] = np.multiply(colorized_skeleton[:,:,0], debug_color[0]/255)
            colorized_skeleton[:,:,1] = np.multiply(colorized_skeleton[:,:,1], debug_color[1]/255)
            colorized_skeleton[:,:,2] = np.multiply(colorized_skeleton[:,:,2], debug_color[2]/255)
            skeletons_image += colorized_skeleton

    PlayMask.save_mask(mask=play_mask, filepath=playmask_path)
    result = PlayMask(
        ball_location=ball_location,
        mask_local_path=playmask_path,
        mask=play_mask
    )

    debug_images.append({'title': 'skeletons', 'img': skeletons_image})
    debug_images.append({'title': 'playmask', 'img': play_mask})
    return (result, debug_images)


def playmask_to_ball_coords(
    playmask: PlayMask,
) -> List[Point]:
    points = []
    playmask_pixels = get_mask_white_pixels(playmask.mask)
    
    for pixel in playmask_pixels:
        x = (pixel[0] - playmask.ball_location.x) / playmask.scale()
        y = (playmask.ball_location.y - pixel[1]) / playmask.scale()
        points.append(Point(x=x, y=y))
    
    return points
