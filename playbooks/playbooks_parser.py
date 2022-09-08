'''
Functions for parsing already scraped play images into usable data.
'''

import sys
from os import path
from typing import List, Dict, Any, Tuple
import traceback
import time
import copy
import json
sys.path.append(path.join(path.dirname(__file__), '..')) # upwards relative imports are hacky

import glog
import cv2
import numpy as np
import skimage
from skimage.morphology import skeletonize

from utils import utils
from utils import visualization_utils as vis_utils
from utils import image_utils
import constants
from plays.play import Play, PlayMask, Point
from playbooks.playbook import Playbook

# For vizualizing skeltons in debug images.
SKELETON_COLORS = [
    [255,0,0],
    [0,255,0],
    [0,0,255],
    [255,255,0],
    [255,0,255],
    [0,255,255],
    [255,255,255],
]

MASK_THRESHOLDS = {
    'regular_route': {
        'min': [190, 170, 75],
        'max': [200, 180, 85]
    },
    'primary_route': {
        'min': [215, 60, 80],
        'max': [225, 70, 90]
    },
    'delayed_route': {
        'min': [0, 0, 240],
        'max': [60, 120, 255]
    },
}

# mask closing params
CLOSING_SIZE = 25
CLOSING_KERNERL = np.ones((CLOSING_SIZE, CLOSING_SIZE), np.uint8)

def _parse_play_image(
    play: Play,
    debug: bool = False,
    verbose: bool = False,
    ) -> Play:
    '''Parse scraped play image into usable data and return as new Play object.'''
    if verbose:
        glog.info(f'parsing play: {play.title()}...')
    parse_start = time.time()
    parsed_play = copy.copy(play)
    try:
        img = cv2.imread(play.image_local_path)
        display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        height, width, _ = display.shape

        debug_images = [
            {'title': play.title(), 'img': display},
        ]

        masks = {}
        for feature_type, mask_thresholds in MASK_THRESHOLDS.items():
            mask = image_utils.img_threshold_by_range(img, min=mask_thresholds['min'], max=mask_thresholds['max'])
            closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSING_KERNERL)
            masks[feature_type] = closed_mask

        ball_location = Point(
            x=width * constants.PLAY_IMAGE_BALL_LOCATION_WIDTH_FRAC,
            y=width * constants.PLAY_IMAGE_BALL_LOCATION_HEIGHT_FRAC,
        )

        play_dir = path.dirname(play.image_local_path)
        play_filename = path.splitext(path.basename(play.image_local_path))[0]
        playmask_path = path.join(play_dir, f'{play_filename}_{constants.PLAYMASK_FILENAME}')
        parsed_playmask, playmask_debug_images = image_utils.parse_playmask_from_masks(
            masks=masks.values(),
            ball_location=ball_location,
            playmask_path=playmask_path
        )
        debug_images += playmask_debug_images

        # sampled_routes = []
        # for scaled_route in scaled_routes:
        #     sampled_route, sample_route_debug_images = image_utils.sample_route_points(
        #         route=scaled_route,
        #         route_scale=constants.PLAYMASK_SCALE
        #     )
        #     sampled_routes.append(sampled_route)
        #     debug_images += sample_route_debug_images

        sampled_playmask = PlayMask.resample(parsed_playmask, constants.PLAYMASK_SCALE)

        # parsed_play.type = # need to determine play type
        parsed_play.playmask = sampled_playmask

        if verbose:
            glog.info(f'..finished parsing {play.title()} in {utils.elapsed_ms(parse_start)}ms: {json.dumps(parsed_play.summary())}')
        if debug:
            vis_utils.display_images(debug_images)
    except Exception as e:
        glog.warning(f'error parsing play: {play.summary()}:\n{traceback.format_exc()}\n{e}\n')
    
    return parsed_play

def parse(playbook_dir: str):
    '''For now access scraped plays via csv and locally downloaded play images'''

    scraped_playbook_filepath = path.join(playbook_dir, constants.SCRAPED_PLAYBOOK_DATA_FILENAME)
    scraped_playbook = Playbook.read_from_json(scraped_playbook_filepath)
    parsed_playbook = copy.copy(scraped_playbook)
    parsed_playbook.plays = []

    parse_start = time.time()
    for scraped_play in scraped_playbook.plays:
        parsed_play = _parse_play_image(scraped_play)
        if parsed_play:
            parsed_playbook.plays.append(parsed_play)

    # DEBUG: play scaling issues
    vis_utils.show_plays(parsed_playbook.plays)

    glog.info(f'successfully parsed {len(parsed_playbook.plays)} / {len(scraped_playbook.plays)} plays from {parsed_playbook.title()} in {utils.elapsed_ms(parse_start)}ms')
    parsed_playbook_filepath = path.join(playbook_dir, constants.PARSED_PLAYBOOK_DATA_FILENAME)
    parsed_playbook.write_to_json(filepath=parsed_playbook_filepath)
