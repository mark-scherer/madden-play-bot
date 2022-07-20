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

CROPPED_SCOREBOARD_COEF = 0.85  # Fraction of image height to keep when cropping out bottom scoreboard

SIDELINE_MASK = {
    'min': [250, 250, 250],
    'max': [255, 255, 255]
}
SIDELINE_EROSION_KERNEL_SIZE = (10, 5)
SIDELINE_TOP_MOE = 0.1  # Margin of error to add when filtering sideline pixels to just those near the ID 'top of the sideline'

LOS_MASK = {
    'min': [0, 30, 80],
    'max': [20, 100, 255]
}
LOS_EROSION_KERNEL_SIZE = (1, 5)

def _find_downfield_corners(input_frame: np.array) ->  Tuple[List[Point], List[Dict[str, Any]]]:
    '''Find & return coordinates for corners of downfield portion of field.

    Notes:
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
    sideline_crop_height = int(height*CROPPED_SCOREBOARD_COEF)
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

def _find_los(input_frame: np.array) ->  int:
    '''Find LOS in frame.
    
    Returns: y coordinate of LOS, in px from top
    '''
    debug_images = []
    height, width, _ = input_frame.shape

    los_crop_height = int(height*CROPPED_SCOREBOARD_COEF)
    los_crop = input_frame.copy()[0:los_crop_height, 0:width]
    
    los_mask = utils.img_threshold_by_range(los_crop, min=LOS_MASK['min'], max=LOS_MASK['max'], reverse=False)
    los_erosion_kernel = np.ones(LOS_EROSION_KERNEL_SIZE, np.uint8)
    cleaned_los_mask = cv2.erode(los_mask, los_erosion_kernel)

    los_pixels = np.where(cleaned_los_mask == [255])
    los = sum(los_pixels[0]) / len(los_pixels[0])
    
    return los


def _dewarp_frame(input_frame: np.array) -> Tuple[np.array, List[Dict[str, Any]]]:
    '''Unwarp frame by converting to orthogonal vertical perspective using a perspective warp.
    
    Returns:
        1. unwarped image
        2. list of debug images
    '''
    debug_images = []
    height, width, _ = input_frame.shape

    # First test by hard-coding transformation vectors
    # Points: [width, height] from TL
        # Must be int32 to be drawn w/ cv2.polyines(), float32 to be input to cv2.getAffineTransform()
    # This works with only 1/2 example frame - need to find transform vectors dynamically
    
    # find downfield
    downfield_corners, field_corner_debug_images = _find_downfield_corners(input_frame=input_frame)
    debug_images += field_corner_debug_images
    downfield_polygon = np.array([
        [pt.x, pt.y] for pt in downfield_corners
    ]).astype(np.float32)
    
    # debug image: id'd downfield region
    downfield_img = input_frame.copy()
    cv2.polylines(downfield_img, [downfield_polygon.astype(np.int32)], False, DEBUG_COLOR, 5)
    debug_images.append({
        'title': 'downfield',
        'img': downfield_img
    })

    # perform perspective transform
    downfield_ymax = max(pt.y for pt in downfield_corners)
    result_polygon = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, downfield_ymax],
        [0, downfield_ymax],
    ]).astype(np.float32)

    warp_matrix = cv2.getPerspectiveTransform(downfield_polygon, result_polygon)
    # result_height = int(height*CROPPED_SCOREBOARD_COEF)
    result = cv2.warpPerspective(input_frame, warp_matrix, (width, height))
    
    debug_images.append({
        'title': 'unwarped test',
        'img': result
    })

    result = input_frame.copy()
    return (result, debug_images)


def parse(scraped_play_animation: PlayAnimation) -> PlayAnimation:
    '''Parse play out of a scraped play animation'''
    input_frame_path = path.join(scraped_play_animation.media_dir, constants.FRAME_FILENAME)
    input_frame = cv2.imread(input_frame_path)
    display_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB) 
    
    dev_images = [{
        'title': 'Input Frame',
        'img': display_frame
    }]

    unwarped_frame, unwarp_dev_images = _dewarp_frame(input_frame=display_frame)
    dev_images += unwarp_dev_images

    # Next crop out deep backfield
        # first find LOS
        # then find scale (px/yd) by finding distance between yardlines
        # then crop height to only include 10yds into backfield

    utils.display_images(images=dev_images)
    return scraped_play_animation