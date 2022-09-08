'''
Script for updating ball template used to find ball in frame.
'''

import sys
from os import path
from typing import Tuple
sys.path.append(path.join(path.dirname(__file__), '..')) # upwards relative imports are hacky
sys.path.append(path.join(path.dirname(__file__), '..', '..', 'utils')) # upwards relative imports are hacky

import glog
import cv2
import numpy as np

import visualization_utils as vis_utils
import image_utils
import constants
import play_animation_scraper

# GIF 1: bills vs chiefs (offense)
# SOURCE_MEDIA_URL = 'https://twitter.com/NextGenStats/status/1485447699600003077'
# MIDDLEMAN_CROP = [0.5, 0.55, 0.6, 0.65]  # [xmin, ymin, xmax, ymax] from top left as fractions
# BALL_CROP_FRACS = [0.52, 0.34, 0.67, 0.55]  # note: these are relative to middleman frame
# NON_BALL_MASKS = {
#     'white': {
#         'min': [180, 180, 180],
#         'max': [255, 255, 255]
#     },
#     'red': {
#         'min': [120, 0, 0],
#         'max': [255, 90, 100]
#     },
#     'light_red': {
#         'min': [165, 100, 95],
#         'max': [255, 180, 190]
#     },
#     'dark_red': {
#         'min': [100, 0, 0],
#         'max': [150, 50, 40]
#     },
#     'green': {
#         'min': [0, 100, 0],
#         'max': [135, 255, 150]
#     },
#     'light_green': {
#         'min': [150, 165, 130],
#         'max': [255, 255, 210]
#     },
#     'dark_green': {
#         'min': [85, 85, 60],
#         'max': [95, 95, 70]
#     },
# }
# BALL_DILATION_1_KERNEL_SIZE = (3, 3)
# BALL_EROSION_KERNEL_SIZE = (3, 3)
# BALL_DILATION_2_KERNEL_SIZE = (1, 1)
# OUTPUT_NAME = 'ball_bills_chiefs.png'


# GIF 2: rams vs bengals (offense)
SOURCE_MEDIA_URL = 'https://twitter.com/NextGenStats/status/1493714866200199172'
MIDDLEMAN_CROP = [0.5, 0.55, 0.6, 0.7]  # [xmin, ymin, xmax, ymax] from top left as fractions
BALL_CROP_FRACS = [0.46, 0.56, 0.66, 0.73]  # note: these are relative to middleman frame
NON_BALL_MASKS = {
    'orange': {
        'min': [150, 50, 10],
        'max': [255, 100, 70]
    },
    'white': {
        'min': [200, 160, 145],
        'max': [255, 255, 255]
    },
    'light_orange': {
        'min': [185, 100, 60],
        'max': [255, 170, 135]
    },
}
BALL_DILATION_1_KERNEL_SIZE = (2, 2)
BALL_EROSION_KERNEL_SIZE = (4, 4)
BALL_DILATION_2_KERNEL_SIZE = (1, 1)
OUTPUT_NAME = 'ball_rams_bengals_0.png'


# GIF 3: rams (offense) vs bengals
# SOURCE_MEDIA_URL = 'https://twitter.com/NextGenStats/status/1493714806162878466'
# MIDDLEMAN_CROP = [0.5, 0.65, 0.6, 0.7]  # [xmin, ymin, xmax, ymax] from top left as fractions
# BALL_CROP_FRACS = [0.53, 0.17, 0.73, 0.72]  # note: these are relative to middleman frame
# NON_BALL_MASKS = {
#     'green': {
#         'min': [0, 130, 70],
#         'max': [125, 255, 155]
#     },
#     'orange': {
#         'min': [155, 50, 0],
#         'max': [255, 125, 80]
#     },
#     'white': {
#         'min': [170, 170, 170],
#         'max': [255, 255, 255]
#     },
#     'light_blue': {
#         'min': [75, 100, 140],
#         'max': [170, 170, 255]
#     },
#     'light_orange': {
#         'min': [230, 140, 110],
#         'max': [255, 185, 160]
#     },
#     'light_green': {
#         'min': [140, 220, 160],
#         'max': [160, 255, 180]
#     },
#     'dark_blue': {
#         'min': [25, 40, 100],
#         'max': [80, 100, 150]
#     },
# }
# BALL_DILATION_1_KERNEL_SIZE = (2, 2)
# BALL_EROSION_KERNEL_SIZE = (3, 3)
# BALL_DILATION_2_KERNEL_SIZE = (1, 1)
# OUTPUT_NAME = 'ball_rams_bengals_1.png'


def main():
    debug_images = []
    
    scraped_play_animation = play_animation_scraper.scrape(url=SOURCE_MEDIA_URL)
    input_frame_path = path.join(scraped_play_animation.dir, constants.FRAME_FILENAME)
    input_frame = cv2.imread(input_frame_path)
    display_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    # debug_images.append({'title': 'input frame', 'img': display_frame})

    middleman_crop, middleman_cropbox = image_utils.crop_image(input=display_frame, crop_fracs=MIDDLEMAN_CROP)
    debug_images.append({'title': 'middleman crop', 'img': middleman_crop})

    ball_crop, ball_cropbox = image_utils.crop_image(input=middleman_crop, crop_fracs=BALL_CROP_FRACS)
    debug_images += [
        {'title': 'ball cropbox', 'img': ball_cropbox},
        {'title': 'ball crop', 'img': ball_crop},
    ]

    ball_mask = np.ones((ball_crop.shape[0], ball_crop.shape[1]), np.uint8)
    for region, mask in NON_BALL_MASKS.items():
        non_ball_mask = image_utils.img_threshold_by_range(ball_crop, min=mask['min'], max=mask['max'], reverse=False)
        other_mask = cv2.bitwise_not(non_ball_mask)
        ball_mask = cv2.bitwise_and(ball_mask, other_mask)
        # debug_images.append({'title': f'non ball mask: {region}', 'img': non_ball_mask})
    
    ball_img = cv2.bitwise_and(ball_crop, ball_crop, mask=ball_mask)
    debug_images += [
        # {'title': 'ball mask', 'img': ball_mask},
        # {'title': 'ball', 'img': ball_img}
    ]

    ball_dilation_1_kernel = np.ones(BALL_DILATION_1_KERNEL_SIZE, np.uint8)
    ball_erosion_kernel = np.ones(BALL_EROSION_KERNEL_SIZE, np.uint8)
    ball_dilation_2_kernel = np.ones(BALL_DILATION_2_KERNEL_SIZE, np.uint8)
    cleaned_ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_DILATE, ball_dilation_1_kernel)
    cleaned_ball_mask = cv2.morphologyEx(cleaned_ball_mask, cv2.MORPH_ERODE, ball_erosion_kernel)
    cleaned_ball_mask = cv2.morphologyEx(cleaned_ball_mask, cv2.MORPH_DILATE, ball_dilation_2_kernel)
    cleaned_ball_img = cv2.bitwise_and(ball_crop, ball_crop, mask=cleaned_ball_mask)
    debug_images += [
        {'title': 'cleaned ball mask', 'img': cleaned_ball_mask},
        {'title': 'cleaned ball', 'img': cleaned_ball_img}
    ]

    output_path = path.join(constants.ASSESTS_DIR, OUTPUT_NAME)
    writeable_ball_img = cv2.cvtColor(cleaned_ball_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, writeable_ball_img)
    glog.info(f'saved ball image to {output_path}')
    vis_utils.display_images(images=debug_images)


main()