'''
Functions for parsing an already scraped play animation.
'''

import sys
from os import path
from typing import List, Any, Tuple, Dict
sys.path.append(path.join(path.dirname(__file__), '..', 'utils')) # upwards relative imports are hacky
sys.path.append(path.join(path.dirname(__file__), '..', 'playbooks')) # upwards relative imports are hacky

import glog
import cv2
import numpy as np

import utils
import constants
from play_animation import PlayAnimation
from playbook import Point

DEBUG_COLOR = [255, 0, 0]

# Generic contants
# Fraction of image height to keep when cropping out bottom scoreboard for both warped and unwarped images
CROPPED_SCOREBOARD_COEF_WARPED = 0.85
CROPPED_SCOREBOARD_COEF_UNWARPED = 0.85
GRASS_MASK = {
    'min': [30, 180, 50],
    'max': [120, 200, 120]
}

# Constants for finding downfield region.
SIDELINE_EROSION_KERNEL_SIZE = (10, 5)
SIDELINE_TOP_MOE = 0.1  # Margin of error to add when filtering sideline pixels to just those near the ID 'top of the sideline'
SIDELINE_MASK = {
    'min': [250, 250, 250],
    'max': [255, 255, 255]
}

# Constants for finding LOS
LOS_MASK = {
    'min': [0, 30, 80],
    'max': [20, 100, 255]
}
LOS_EROSION_KERNEL_SIZE = (1, 5)

# Constants for finding field scale
FIELD_WIDTH_YDS = 160/3  # 160' wide
# dimensions of crop when counting sideline hash marks
YSCALE_CROP_YMIN = 0.4  # need to avoid possible endzone
YSCALE_CROP_YMAX = 0.7  # need to avoid black region at bottom edges of frame
YSCALE_CROP_WIDTH = 0.05
HASH_CLOSING_KERNEL_SIZE = (5, 5)

# other constants
BACKFIELD_YDS_TO_INCLUDE = 10  # yds behind LOS to include in pre-processed frame

def _find_downfield_corners(input_frame: np.array) ->  Tuple[List[Point], List[Dict[str, Any]]]:
    '''Find & return coordinates for corners of downfield portion of field.

    Notes:
    - Uses pretty specialized logic to find sideline bounds
    - Downfield: past the LOS thru the back of the endzone
        - bottom border is not really LOS, it's yardline when sidelines intercept side edges of the image
    - Coordinate system 0,0 at TL of image

    Returns:
        1. List of corner points
        2. List of debug images
    '''
    debug_images = []
    height, width, _ = input_frame.shape

    # Crop to just image top right
    # TR not TL b/c sidelines are symmetrical and TL has interference from logo
    sideline_crop_width = int(0.5*width)
    sideline_crop_height = int(height*CROPPED_SCOREBOARD_COEF_WARPED)
    sideline_crop = input_frame.copy()[0:sideline_crop_height, sideline_crop_width:width-1]

    sideline_mask = utils.img_threshold_by_range(sideline_crop, min=SIDELINE_MASK['min'], max=SIDELINE_MASK['max'], reverse=False)

    sideline_erosion_kernel = np.ones(SIDELINE_EROSION_KERNEL_SIZE, np.uint8)
    cleaned_sideline_mask = cv2.morphologyEx(sideline_mask, cv2.MORPH_OPEN, sideline_erosion_kernel)

    # Downstream logic for needs to know if back endzone sideline is visible
    back_sideline_crop_width = int(0.5*sideline_crop_width)
    back_sideline_crop = cleaned_sideline_mask[:,0:back_sideline_crop_width]
    back_sideline_crop_sideline_pixels = utils.get_mask_white_pixels(mask=back_sideline_crop)
    back_sideline_visible = back_sideline_crop_sideline_pixels.shape[0] > 0

    sideline_pixels = utils.get_mask_white_pixels(mask=cleaned_sideline_mask)

    # Now that sideline isolated first find ymin and ymax coordinates
        # ymin will be top of sideline
        # ymax will be where sideline intersect image's right edge
    # Remember top of image is y = 0
    _, _, sideline_ymin, sideline_ymax = utils.get_pixel_extremes(pixels=sideline_pixels)
    
    # We now know downfield's BR corner (intersection of sideline with right edge of image)
    sideline_bottom_inside_corner = Point(x=sideline_crop_width-1, y=sideline_ymax)
    downfield_bot_right_corner = Point(
        x = sideline_bottom_inside_corner.x + sideline_crop_width,  # adjust for fact was calculated within sideline_crop not original image
        y = sideline_bottom_inside_corner.y
    )

    # Logic for finding downfield's TR corner depends on if back of endzone sideline is visible
    downfield_top_right_corner = None
    if back_sideline_visible:
        # First find TR corner of sideline by isolating just the top slice and finding its xmax
        adj_sideline_ymin = max(sideline_ymin, 1)  # makes logic easier if ymin > 0
        sideline_top_pixels = sideline_pixels[sideline_pixels[:,1] < adj_sideline_ymin * (1 + SIDELINE_TOP_MOE)]
        _, sideline_top_xmax, _, _ = utils.get_pixel_extremes(pixels=sideline_top_pixels)
        sideline_top_right_corner = Point(x=sideline_top_xmax, y=sideline_ymin)
        
        # Need to convert back sideline TR corner to downfield TR by removing the width of the sideline
        # First must find the width of the sideline by looking at just its left edge
        _, _, back_sideline_ymin, back_sideline_ymax = utils.get_pixel_extremes(pixels=back_sideline_crop_sideline_pixels)
        sideline_thickness = back_sideline_ymax - back_sideline_ymin
        
        # Now we know downfield TR corner by estimating the inside TR corner of the sideline
        sideline_inside_corner = Point(
            x = sideline_top_right_corner.x - sideline_thickness,
            y = sideline_top_right_corner.y + sideline_thickness
        )
        downfield_top_right_corner = Point(
        x = sideline_inside_corner.x + sideline_crop_width,  # adjust for fact was calculated within sideline_crop not original image
        y = sideline_inside_corner.y
    )
    else:
        # If back sideline not visible, just take sideline xmin
        sideline_xmin_idx = np.argmin(sideline_pixels, axis=0)[0]
        sideline_xmin = sideline_pixels[sideline_xmin_idx][0]
        
        # Now know downfield's TR corner (intersection of sideline and top edge of image)
        sideline_top_inside_corner = Point(x=sideline_xmin, y=0)
        downfield_top_right_corner = Point(
            x = sideline_top_inside_corner.x + sideline_crop_width, # adjust for fact was calculated within sideline_crop not original image
            y = sideline_top_inside_corner.y
        )

    # Now mirror downfield right corners to find left corners
    downfield_top_left_corner = Point(
        x = width - downfield_top_right_corner.x,
        y = downfield_top_right_corner.y
    )
    downfield_bot_left_corner = Point(
        x = width - downfield_bot_right_corner.x,
        y = downfield_bot_right_corner.y
    )

    downfield_corners = [
        downfield_top_left_corner,
        downfield_top_right_corner,
        downfield_bot_right_corner,
        downfield_bot_left_corner
    ]

    return (downfield_corners, debug_images)

def _find_field_extremes(frame: np.array) -> Tuple[Tuple[int, int, int, int], List[np.array]]:
    '''Find extreme values in x & y for field region in given frame.

    Notes:
    - uses simple method of taking mask of grass in image

    Return:
        1. tuple of xmin, xmax, ymin, ymax
        2. list of debug images
    '''
    debug_images = []
    
    grass_mask = utils.img_threshold_by_range(img=frame, min=GRASS_MASK['min'], max=GRASS_MASK['max'], reverse=False)
    debug_images.append({
        'title': 'grass mask',
        'img': grass_mask
    })

    grass_pixels = utils.get_mask_white_pixels(mask=grass_mask)
    return (utils.get_pixel_extremes(pixels=grass_pixels), debug_images)


def _find_yscale(frame: np.array) -> Tuple[float, List[np.array]]:
    '''Find vertical image scale in px/yd by counting sideline hash marks. 
    
    Args:
        1. frame: orthogonal image cropped to just field region

    Return: 
        1. vertical image scale in px/yd
        2. list of debug images
    '''
    height, width, _ = frame.shape
    debug_images = []

    # crop to TR to get clean sideline while avoiding logo
    yscale_crop_ymin = int(height*YSCALE_CROP_YMIN)
    yscale_crop_ymax = int(height*YSCALE_CROP_YMAX)
    yscale_crop_width = int(height*YSCALE_CROP_WIDTH)
    yscale_crop = frame[yscale_crop_ymin:yscale_crop_ymax, (width - yscale_crop_width):(width - 1)]
    debug_images.append({
        'title': 'yscale crop',
        'img': yscale_crop
    })

    grass_mask = utils.img_threshold_by_range(img=yscale_crop, min=GRASS_MASK['min'], max=GRASS_MASK['max'], reverse=False)
    debug_images.append({
        'title': 'grass mask',
        'img': grass_mask
    })

    hash_mask = cv2.bitwise_not(grass_mask)
    debug_images.append({
        'title': 'hash mask',
        'img': hash_mask
    })

    hash_closing_kernel = np.ones(HASH_CLOSING_KERNEL_SIZE, np.uint8)
    cleaned_hash_mask = cv2.morphologyEx(hash_mask, cv2.MORPH_CLOSE, hash_closing_kernel)
    cleaned_hash_mask_height = cleaned_hash_mask.shape[0]
    debug_images.append({
        'title': 'cleaned hash mask',
        'img': cleaned_hash_mask
    })

    hash_contours, _ = cv2.findContours(cleaned_hash_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hash_contours_img = np.zeros(cleaned_hash_mask.shape, np.uint8)
    hash_contours_img = cv2.drawContours(hash_contours_img, hash_contours, -1, DEBUG_COLOR, 1)
    debug_images.append({
        'title': 'hash contours',
        'img': hash_contours_img
    })

    yscale = cleaned_hash_mask_height / len(hash_contours)
    return (yscale, debug_images)


def _find_los(frame: np.array) ->  Tuple[int, np.array]:
    '''Find LOS in frame.
    
    Returns: 
        1. y coordinate of LOS, in px from top
        2. list of debug images
    '''
    debug_images = []
    height, width, _ = frame.shape

    los_crop_height = int(height*CROPPED_SCOREBOARD_COEF_UNWARPED)
    los_crop = frame.copy()[0:los_crop_height, 0:width]
    debug_images.append({
        'title': 'los crop',
        'img': los_crop
    })
    
    los_mask = utils.img_threshold_by_range(los_crop, min=LOS_MASK['min'], max=LOS_MASK['max'], reverse=False)
    debug_images.append({
        'title': 'los mask',
        'img': los_mask
    })

    los_erosion_kernel = np.ones(LOS_EROSION_KERNEL_SIZE, np.uint8)
    cleaned_los_mask = cv2.erode(los_mask, los_erosion_kernel)
    debug_images.append({
        'title': 'cleaned los mask',
        'img': cleaned_los_mask
    })

    los_pixels = np.where(cleaned_los_mask == [255])
    los = int(sum(los_pixels[0]) / len(los_pixels[0]))
    
    return (los, debug_images)


def _unwarp_frame(input_frame: np.array) -> Tuple[np.array, List[Dict[str, Any]]]:
    '''Unwarp frame by converting to orthogonal vertical perspective using a perspective warp.
    
    Returns:
        1. unwarped image
        2. field scale in unwarped image (px/yd)
        3. list of debug images
    '''
    debug_images = []
    height, width, _ = input_frame.shape
    
    # find downfield region
    downfield_corners, field_corner_debug_images = _find_downfield_corners(input_frame=input_frame)
    debug_images += field_corner_debug_images
    downfield_polygon = np.array([
        [pt.x, pt.y] for pt in downfield_corners
    ]).astype(np.float32)
    downfield_img = input_frame.copy()
    cv2.polylines(downfield_img, [downfield_polygon.astype(np.int32)], False, DEBUG_COLOR, 5)
    # debug_images.append({
    #     'title': 'downfield',
    #     'img': downfield_img
    # })

    # perform perspective transform
    downfield_ymax = max(pt.y for pt in downfield_corners)
    result_polygon = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, downfield_ymax],
        [0, downfield_ymax],
    ]).astype(np.float32)
    warp_matrix = cv2.getPerspectiveTransform(downfield_polygon, result_polygon)
    unwarped_frame = cv2.warpPerspective(input_frame, warp_matrix, (width, height))

    # crop to just field region
    field_extremes, field_extremes_debug_images = _find_field_extremes(frame=unwarped_frame)
    field_xmin, field_xmax, field_ymin, field_ymax = field_extremes
    cropped_unwarped_frame = unwarped_frame[field_ymin:field_ymax, field_xmin:field_xmax]
    # debug_images += field_extremes_debug_images
    # debug_images.append({
    #     'title': 'cropped unwarped',
    #     'img': cropped_unwarped_frame
    # })

    # rescale image so y & x scales equal
    cropped_height, cropped_width = cropped_unwarped_frame.shape[0:2]
    xscale = cropped_width / FIELD_WIDTH_YDS
    yscale, yscale_debug_images = _find_yscale(frame=cropped_unwarped_frame)
    rescaled_unwarped_frame = cropped_unwarped_frame.copy()
    rescaled_height = int(cropped_height * (xscale/ yscale))  # keep height constant and adjust width
    rescaled_shape = (cropped_width, rescaled_height)
    rescaled_unwarped_frame = cv2.resize(cropped_unwarped_frame, rescaled_shape)
    # debug_images += yscale_debug_images

    return (rescaled_unwarped_frame, xscale, debug_images)

def preprocess_frame(display_frame: np.array) -> Tuple[np.array, float, int, List[np.array]]:
    '''Prepare frame for parsing. Performs:
    1. unwarps to orthogonal view
    2. crops to just field region
    3. rescales to equal x & y by yardage
    4. crops further to just 10yds into backfield

    Args:
        1. display_frame: original frame from play animation mapped to RGB
    
    Return:
        1. processed frame
        2. field scale in processed frame (yds/pixels)
        3. LOS y value in processed frame (top = 0)
        4. list of debug images
    '''
    debug_images = []

    unwarped_frame, field_scale, unwarp_dev_images = _unwarp_frame(input_frame=display_frame)
    debug_images += unwarp_dev_images
    # debug_images.append({
    #     'title': 'unwarped',
    #     'img': unwarped_frame
    # })

    # Next crop out deep backfield
    los_y, los_dev_images = _find_los(frame=unwarped_frame)
    # debug_images += los_dev_images

    unwarped_height, unwarped_width = unwarped_frame.shape[0:2]
    backfield_ymax = los_y + int(BACKFIELD_YDS_TO_INCLUDE*field_scale)

    processed_height = min(backfield_ymax, unwarped_height)
    processed_frame = unwarped_frame[0:processed_height, :]

    return (processed_frame, field_scale, los_y, debug_images)


def parse(scraped_play_animation: PlayAnimation) -> PlayAnimation:
    '''Parse play out of a scraped play animation'''
    input_frame_path = path.join(scraped_play_animation.media_dir, constants.FRAME_FILENAME)
    input_frame = cv2.imread(input_frame_path)
    display_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB) 
    debug_images = [{
        'title': 'Input Frame',
        'img': display_frame
    }]
    
    processed_frame, field_scale, los_y, preprocess_debug_images = preprocess_frame(display_frame=display_frame)
    # debug_images += preprocess_debug_images
    debug_images.append({
        'title': 'processed',
        'img': processed_frame
    })

    utils.display_images(images=debug_images)
    return scraped_play_animation