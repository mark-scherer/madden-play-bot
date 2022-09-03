'''
Various utils.
'''

import sys
from os import path
from typing import List, Dict, Any, Tuple, Union
import math
sys.path.append(path.join(path.dirname(__file__), '..')) # upwards relative imports are hacky

import cv2
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
import colorsys
import skimage
from skimage.morphology import skeletonize

import constants
from playbook import Play, Route, Point


MIN_ROUTE_POINTS = 10  # minimum pixels to parse a route

PLAY_BACKFIELD_YARDS = 10
PLAY_DOWNFIELD_YARDS = 40
FIELD_WIDTH_YARDS = 160/3
YARDLINE_COLOR = 'lightgrey'
YARDLINE_THICKNESS_PTS = 2
SIDELINE_THICKNESS_YARDS = 2
HASMARKS_INSIDE_WIDTH_YARDS = 18.6/3
HASHMARKS_LENGTH_YARDS = 2/3
YARD_MARKS_CENTER_DISTANCE_TO_SIDELINE_YARDS = (12 + 9) / 2

PLAY_COLOR = 'black'
BALL_SIZE_PTS = 4

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

def rgb_to_hsv(rgb: Tuple[int, int, int], normalize: bool = True) -> Tuple[int, int, int]:
    '''Converts rgb color to hsv.
    
    Args:
        1. rgb: rgb color as tuple of uint8
        2. normalize: return resulting hsv values 0-1? otherwise 0-255

    Return: hsv color as tuple (normalize = True: 0-1, otherwise uint8)
    '''
    hsv = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
    converted_hsv = hsv if normalize else (
        int(hsv[0]*255),
        int(hsv[1]*255),
        int(hsv[2]*255)
    )
    return converted_hsv

def hsv_to_rgb(
    hsv: Union[Tuple[int, int, int], Tuple[float, float, float]],
    normalized: bool = True
) -> Tuple[int, int, int]:
    '''Converts hsv color to rgb.
    
    Args:
        1. hsv: hsv color as tuple of floats 0-1 (normalized = True) or uint8
        2. normalized: hsv arg normalized as float 0-1? Otherwise uint8

    Return: hsv color as tuple (normalize = True: 0-1, otherwise uint8)
    '''
    assert normalized, 'non-normalized hsv input not yet implemented'

    rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])
    return (
        int(rgb[0]*255),
        int(rgb[1]*255),
        int(rgb[2]*255)
    )

def rgb_to_hsl(rgb: Tuple[int, int, int], normalize: bool = True) -> Tuple[int, int, int]:
    '''Converts rgb color to hsl.
    
    Args:
        1. rgb: rgb color as tuple of uint8
        2. normalize: return resulting hsv values 0-1? otherwise 0-255

    Return: hsv color as tuple (normalize = True: 0-1, otherwise uint8)
    '''
    assert normalize, 'non-normalized hsv input not yet implemented'
    
    hls = colorsys.rgb_to_hls(rgb[0]/255, rgb[1]/255, rgb[2]/255)
    converted_hsl = (
        hls[0],
        hls[2],
        hls[1]
    )
    return converted_hsl

def overlap_semitransparent_color(
    background_color: Tuple[int, int, int],
    foreground_color: Tuple[int, int, int],
    foreground_opacity: float,
    cast_to_int: bool = True
) -> Tuple[int, int, int]:
    '''Return equivalent opaque color after overlaying semitransparent foreground color over opaque background color.'''
    
    def _overlay_channel(background_value: int, foreground_value: int, opacity: float, cast_to_int: bool) -> int:
        overlaid = (opacity*foreground_value) + ((1 - opacity)*background_value)
        return int(overlaid) if cast_to_int else overlaid
    
    assert foreground_opacity >= 0 and foreground_opacity <= 1, \
        'opacity must be 0-1'

    return (
        _overlay_channel(background_color[0], foreground_color[0], foreground_opacity, cast_to_int),
        _overlay_channel(background_color[1], foreground_color[1], foreground_opacity, cast_to_int),
        _overlay_channel(background_color[2], foreground_color[2], foreground_opacity, cast_to_int),
    )
    

def clamp(input, _min=0, _max=1):
    '''Clamps input between min and max.
    
    This really isn't built in to python?
    '''
    return max(_min, min(_max, input))


def parse_routes_from_masks(masks: List[np.array]) -> Tuple[List[Route], List[np.array]]:
    '''Given a list of cleaned masks parse into individual routes.
    
    Return:
        1. list of routes
        2. list of debug images
    '''
    assert len(masks) > 0
    mask_shape = masks[0].shape

    debug_images = []
    skeletons_image = np.zeros([mask_shape[0], mask_shape[1], 3], np.uint8)

    routes = []
    contour_idx = -1
    for i, mask in enumerate(masks):
        contours, _ = cv2.findContours(mask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        all_mask_contours_image = np.zeros(mask_shape, np.uint8)
        all_mask_contours_image = cv2.drawContours(all_mask_contours_image, contours, -1, constants.DEBUG_COLOR, -1)
        # debug_images.append({'title': f'all contour {i}', 'img': all_mask_contours_image})
        
        for j, contour in enumerate(contours):
            contour_idx += 1
            color_idx = contour_idx % len(constants.ROUTE_COLORS)
            debug_color = constants.ROUTE_COLORS[color_idx]
            
            contour_image = np.zeros(mask_shape, np.uint8)
            contour_image = cv2.drawContours(contour_image, [contour], 0, (255, 255, 255), -1)
            # debug_images.append({'title': f'contour {contour_idx}', 'img': contour_image.copy()})

            skeletonized = skeletonize(skimage.img_as_float(contour_image.copy())).astype('uint8') * 255
            route_points = np.argwhere(skeletonized == 255)
            if len(route_points) < MIN_ROUTE_POINTS:
                continue

            colorized_skeleton = cv2.cvtColor(skeletonized, cv2.COLOR_GRAY2RGB)
            colorized_skeleton[:,:,0] = np.multiply(colorized_skeleton[:,:,0], debug_color[0]/255)
            colorized_skeleton[:,:,1] = np.multiply(colorized_skeleton[:,:,1], debug_color[1]/255)
            colorized_skeleton[:,:,2] = np.multiply(colorized_skeleton[:,:,2], debug_color[2]/255)
            skeletons_image += colorized_skeleton
            
            routes.append(Route(
                points=[Point(pt[1], pt[0]) for pt in route_points]
            ))

    debug_images.append({'title': 'skeletons', 'img': skeletons_image})
    
    return (routes, debug_images)

def show_play(play: Play) -> None:
    '''Display play in a popup.'''
    field_xmin = -1 * FIELD_WIDTH_YARDS / 2
    field_xmax = FIELD_WIDTH_YARDS / 2
    field_ymin = -1*PLAY_BACKFIELD_YARDS
    field_ymax = PLAY_DOWNFIELD_YARDS
    
    # set title and axis limits and clear markers
    fig, ax = plt.subplots()
    ax.set_xlim(field_xmin, field_xmax)
    ax.set_ylim(field_ymin, field_ymax)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(play.title())

    # Draw yard lines
    yard_lines_y = range(field_ymin, field_ymax + 1, 5)
    ax.hlines(y=yard_lines_y, xmin=field_xmin, xmax=field_xmax, color=YARDLINE_COLOR, linewidth=YARDLINE_THICKNESS_PTS)

    # Draw sidelines
    sideline_height = field_ymax - field_ymin
    left_sideline = patches.Rectangle(
        xy=(field_xmin - SIDELINE_THICKNESS_YARDS, field_ymin),
        width=SIDELINE_THICKNESS_YARDS,
        height=sideline_height,
        color=YARDLINE_COLOR    
    )
    right_sideline = patches.Rectangle(
        xy=(field_xmax, field_ymin),
        width=SIDELINE_THICKNESS_YARDS,
        height=sideline_height,
        color=YARDLINE_COLOR    
    )
    ax.add_patch(left_sideline)
    ax.add_patch(right_sideline)

    # Draw hashes
    hashmarks_xmin = HASMARKS_INSIDE_WIDTH_YARDS/2
    hashmarks_xmax = hashmarks_xmin + HASHMARKS_LENGTH_YARDS
    hashmarks_y = range(field_ymin, field_ymax + 1, 1)
    for y in hashmarks_y:
        ax.hlines(y=y, xmin=hashmarks_xmin, xmax=hashmarks_xmax, color=YARDLINE_COLOR, linewidth=YARDLINE_THICKNESS_PTS)
        ax.hlines(y=y, xmin=-1*hashmarks_xmin, xmax=-1*hashmarks_xmax, color=YARDLINE_COLOR, linewidth=YARDLINE_THICKNESS_PTS)

    # Draw numbers
    yard_markers_y = range(field_ymin, field_ymax + 1, 10)
    yard_markers_x = FIELD_WIDTH_YARDS/2 - YARD_MARKS_CENTER_DISTANCE_TO_SIDELINE_YARDS
    for y in yard_markers_y:
        ax.text(x=yard_markers_x, y=y, s=y, 
            ha='center', va='center', color=YARDLINE_COLOR,
            rotation='vertical', size='x-large')
        ax.text(x=-1*yard_markers_x, y=y, s=y, 
            ha='center', va='center', color=YARDLINE_COLOR,
            rotation='vertical', size='x-large')

    # Draw ball
    ax.plot(0, 0, 's', color=PLAY_COLOR, ms=BALL_SIZE_PTS)

    # Draw routes
    for route in play.routes:
        pt_x = [pt.x for pt in route.points]
        pt_y = [pt.y for pt in route.points]
        ax.plot(pt_x, pt_y, color=PLAY_COLOR, marker='.', linestyle='none')

    plt.show()