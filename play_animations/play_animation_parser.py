'''
Functions for parsing an already scraped play animation.
'''

import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '..', 'utils')) # upwards relative imports are hacky
sys.path.append(path.join(path.dirname(__file__), '..', 'playbooks')) # upwards relative imports are hacky

import glog
import cv2
import numpy as np

import utils
import constants
import play_animation_preprocessor as preprocessor
from play_animation import PlayAnimation

DEBUG_COLOR = [255, 0, 0]

def parse(scraped_play_animation: PlayAnimation) -> PlayAnimation:
    '''Parse play out of a scraped play animation'''
    input_frame_path = path.join(scraped_play_animation.media_dir, constants.FRAME_FILENAME)
    input_frame = cv2.imread(input_frame_path)
    display_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB) 
    debug_images = [{
        'title': 'input frame',
        'img': display_frame
    }]
    
    processed_frame, field_scale, los_y, preprocess_debug_images = preprocessor.preprocess_frame(display_frame=display_frame)
    # debug_images += preprocess_debug_images
    debug_images.append({
        'title': 'processed',
        'img': processed_frame
    })

    utils.display_images(images=debug_images)
    return scraped_play_animation