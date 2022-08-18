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
from playbook import Point

SCOREBOARD_COLORS_CROP_WIDTH = 0.5

FIELD_MARKS_MASK = {
    'min': [150, 150, 150],
    'max': [255, 255, 255]
}
NONSCOREBOARD_DILATION_KERNEL_SIZE = (5, 5)

# Oline crop for determining offsense color.
    # Need to keep tight to just center to prevent intrusion of grass, midfield logo, yardlines, etc.
OLINE_CROP_YOFFSET = 10
OLINE_CROP_HEIGHT = 5  # narrow b/c sometimes center is up on ball, sometime back a bit
OLINE_CROP_WIDTH = 25
TEAM_COLOR_H_THRESHOLD = 0.1  # 0-1, count oline pixel hue as either team hue if within this threshold
TEAM_COLOR_S_THRESHOLD = 1
TEAM_COLOR_V_THRESHOLD = 0.3

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

    # Just take two most common colors but first convert to HSV to allow rounding
    team_colors_pixels = utils.get_mask_white_pixels(mask=team_colors_mask)
    color_counter = {}
    for pixel in team_colors_pixels:
        rgb_color = tuple(team_colors_frame[pixel[1], pixel[0], :])
        hsv_color = utils.rgb_to_hsv(rgb_color)
        cleaned_hsv_color = (
            round(hsv_color[0], 2),  # hue
            round(hsv_color[1], 1),  # saturation
            round(hsv_color[2], 1),  # value
        )
        if cleaned_hsv_color not in color_counter:
            color_counter[cleaned_hsv_color] = 0
        color_counter[cleaned_hsv_color] += 1
    sorted_color_counter = sorted(color_counter.items(), key=lambda item: item[1], reverse=True)
    
    team_color_1 = utils.hsv_to_rgb(sorted_color_counter[0][0])
    team_color_2 = utils.hsv_to_rgb(sorted_color_counter[1][0])
    return (team_color_1, team_color_2, debug_images)

def _get_offense_color(
    display_frame: np.array,
    preprocessed_frame: np.array,
    ball_location: Point) -> Tuple[Tuple[int], List[np.array]]:
    '''Parse offensive color.
    
    Args:
        display_frame: original frame from play animation
        preprocessed_frame: dewarped & cropped frame
        ball_location: location of ball before the snap in preprocessed_frame

    Return:
        1. color of offense
        2. list of debug images
    '''
    debug_images = []
    
    # First get team colors.
    team_color_1, team_color_2, team_colors_debug_images = _get_team_colors(display_frame=display_frame)
    debug_images += team_colors_debug_images

    # Then crop to just oline and see which team color more prevalent
    oline_crop_left = int(ball_location.x - (OLINE_CROP_WIDTH/2))
    oline_crop_top = ball_location.y + OLINE_CROP_YOFFSET
    oline_crop = preprocessed_frame[
        oline_crop_top : oline_crop_top + OLINE_CROP_HEIGHT,
        oline_crop_left : oline_crop_left + OLINE_CROP_WIDTH    
    ]
    debug_images.append({'title': 'oline crop', 'img': oline_crop})

    hsv_1 = utils.rgb_to_hsv(team_color_1)
    hsv_2 = utils.rgb_to_hsv(team_color_2)
    hsv_1_count = 0
    hsv_2_count = 0

    for i in range(OLINE_CROP_WIDTH):
        for j in range(OLINE_CROP_HEIGHT):

            # convert to hsv for better rounded comparison
            pixel_color = oline_crop[j, i, :]
            pixel_hsv = utils.rgb_to_hsv(pixel_color)

            if abs(pixel_hsv[0] - hsv_1[0]) < TEAM_COLOR_H_THRESHOLD and \
                abs(pixel_hsv[1] - hsv_1[1]) < TEAM_COLOR_S_THRESHOLD and \
                abs(pixel_hsv[2] - hsv_1[2]) < TEAM_COLOR_V_THRESHOLD:
                hsv_1_count += 1
            if abs(pixel_hsv[0] - hsv_2[0]) < TEAM_COLOR_H_THRESHOLD and \
                abs(pixel_hsv[1] - hsv_2[1]) < TEAM_COLOR_S_THRESHOLD and \
                abs(pixel_hsv[2] - hsv_2[2]) < TEAM_COLOR_V_THRESHOLD:
                hsv_2_count += 1

    offense_color = team_color_1 if hsv_1_count > hsv_2_count else team_color_2
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
    
    preprocessed_frame, field_scale, ball_location, preprocess_debug_images = preprocessor.preprocess_frame(display_frame=display_frame)
    debug_images += preprocess_debug_images
    debug_images.append({
        'title': 'preprocessed',
        'img': preprocessed_frame
    })

    offense_color, offense_color_debug_images = _get_offense_color(
        display_frame=display_frame,
        preprocessed_frame=preprocessed_frame,
        ball_location=ball_location)
    # debug_images += offense_color_debug_images

    utils.display_images(images=debug_images)
    return scraped_play_animation