'''
Functions for parsing an already scraped play animation.
'''

import sys
from os import path
from typing import Tuple, List
sys.path.append(path.join(path.dirname(__file__), '..', 'utils')) # upwards relative imports are hacky
sys.path.append(path.join(path.dirname(__file__), '..', 'playbooks')) # upwards relative imports are hacky

import glog
import cv2
import numpy as np

import utils
import constants
import play_animation_preprocessor as preprocessor
from play_animation import PlayAnimation

SCOREBOARD_COLORS_CROP_WIDTH = 0.5

FIELD_MARKS_MASK = {
    'min': [150, 150, 150],
    'max': [255, 255, 255]
}
NONSCOREBOARD_DILATION_KERNEL_SIZE = (5, 5)

def _get_team_colors(display_frame: np.array) -> Tuple[Tuple[int], Tuple[int], List[np.array]]:
    '''Get colors for both teams.
    
    Args:
        display_frame: original frame from play animation
        preprocessed_frame: dewarped & cropped frame

    Return:
        1. color team 1
        1. color team 2
        2. list of debug images
    '''
    debug_images = []
    input_height, input_width, _ = display_frame.shape

    # Crop to just left half of scorboard to get color block for each team
    scoreboard_crop_height = int(input_height*constants.CROPPED_SCOREBOARD_COEF_WARPED)
    scoreboard_crop_width = int(input_width*SCOREBOARD_COLORS_CROP_WIDTH)
    scoreboard_crop = display_frame[scoreboard_crop_height:input_height-1, 0:scoreboard_crop_width]
    debug_images.append({
        'title': 'scoreboard_crop',
        'img': scoreboard_crop
    })

    # Filter out grass & field markings to just get colored scoreboard elements
    grass_mask = utils.img_threshold_by_range(img=scoreboard_crop, max=constants.GRASS_MASK['max'], min=constants.GRASS_MASK['min'], reverse=False)
    white_mask = utils.img_threshold_by_range(img=scoreboard_crop, max=FIELD_MARKS_MASK['max'], min=FIELD_MARKS_MASK['min'], reverse=False)
    nonscoreboard_dilation_kernel = np.ones(NONSCOREBOARD_DILATION_KERNEL_SIZE, np.uint8)
    cleaned_grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_DILATE, nonscoreboard_dilation_kernel)
    cleaned_white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_DILATE, nonscoreboard_dilation_kernel)
    team_colors_mask = cv2.bitwise_not(cv2.bitwise_or(cleaned_grass_mask, cleaned_white_mask))
    team_colors_frame = cv2.bitwise_and(scoreboard_crop, scoreboard_crop, mask=team_colors_mask)
    debug_images.append({
        'title': 'team colors mask',
        'img': team_colors_mask
    })
    debug_images.append({
        'title': 'team colors frame',
        'img': team_colors_frame
    })

    # Just take two most common colors
    team_colors_pixels = utils.get_mask_white_pixels(mask=team_colors_mask)
    color_counter = {}
    for pixel in team_colors_pixels:
        color = tuple(team_colors_frame[pixel[1], pixel[0], :])
        if color not in color_counter:
            color_counter[color] = 0
        color_counter[color] += 1
    sorted_color_counter = sorted(color_counter.items(), key=lambda item: item[1], reverse=True)
    
    team_color_1 = sorted_color_counter[0][0]
    team_color_2 = sorted_color_counter[1][0]
    return (team_color_1, team_color_2, debug_images)

def _get_offense_color(
    display_frame: np.array,
    preprocessed_frame: np.array,
    los_y: int) -> Tuple[Tuple[int], List[np.array]]:
    '''Parse offensive color.
    
    Args:
        display_frame: original frame from play animation
        preprocessed_frame: dewarped & cropped frame
        los_y: y value of LOS in preprocessed_frame

    Return:
        1. color of offense
        2. list of debug images
    '''
    debug_images = []
    
    team_color_1, team_color_2, team_colors_debug_images = _get_team_colors(display_frame=display_frame)
    # debug_images += team_colors_debug_images

    # need to actually find offense color
    offense_color = team_color_1
    return (offense_color, debug_images)


def parse(scraped_play_animation: PlayAnimation) -> PlayAnimation:
    '''Parse play out of a scraped play animation'''
    input_frame_path = path.join(scraped_play_animation.media_dir, constants.FRAME_FILENAME)
    input_frame = cv2.imread(input_frame_path)
    display_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB) 
    debug_images = [{
        'title': 'input frame',
        'img': display_frame
    }]

    # NOTE: think preprocessing should find LOS by finding ball... current method can get confused by blue elements on the field
        # See example play animation with issue
    # then _get_offense_color() can see which team color more common in OLine box instead of behind LOS
        # Behind LOS liable to be confused by onfield elements
    
    preprocessed_frame, field_scale, ball_location, preprocess_debug_images = preprocessor.preprocess_frame(display_frame=display_frame)
    debug_images += preprocess_debug_images
    debug_images.append({
        'title': 'preprocessed',
        'img': preprocessed_frame
    })

    offense_color, offense_color_debug_images = _get_offense_color(
        display_frame=display_frame,
        preprocessed_frame=preprocessed_frame,
        los_y=ball_location.y)
    # debug_images += offense_color_debug_images

    utils.display_images(images=debug_images)
    return scraped_play_animation