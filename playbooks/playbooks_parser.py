'''
Functions for parsing already scraped play images into usable data.
'''

import sys
from typing import List, Dict, Any, Tuple
import traceback
import time
import copy
import json
sys.path.append(path.join(path.dirname(__file__), '..', 'utils')) # upwards relative imports are hacky

import glog
import cv2
import numpy as np
import skimage
from skimage.morphology import skeletonize

from ..utils import utils
import constants
from playbook import Playbook, Play, Route, Point

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
        # grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        debug_images = [
            {'title': play.title(), 'img': display},
        ]

        skeletons_image = np.zeros(img.shape, np.uint8)
        parsed_routes = []
        for feature_type, mask_thresholds in MASK_THRESHOLDS.items():
            mask = utils.img_threshold_by_range(img, min=mask_thresholds['min'], max=mask_thresholds['max'])
            closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSING_KERNERL)

            # debug_images += [
            #     {'title': f'{feature_type}: mask', 'img': mask.copy()},
            #     {'title': f'{feature_type}: closed_mask', 'img': closed_mask.copy()},
            # ]

            # separate each route via contours
            contours, _ = cv2.findContours(closed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)            
            for i, contour in enumerate(contours):
                contour_image = np.zeros(img.shape[:2], np.uint8)
                contour_image = cv2.drawContours(contour_image, contours, i, (255,255,255), -1)
                skeletonized = skeletonize(skimage.img_as_float(contour_image)).astype('uint8') * 255
                skeleton_color = SKELETON_COLORS[len(parsed_routes)]
                
                route_points = np.argwhere(skeletonized == 255)
                route = Route(
                    points=[Point(int(pt[0]), int(pt[1])) for pt in route_points]
                )
                parsed_routes.append(route)

                colorized_skeleton = cv2.cvtColor(skeletonized,cv2.COLOR_GRAY2RGB)
                colorized_skeleton[:,:,0] = np.multiply(colorized_skeleton[:,:,0], skeleton_color[0]/255)
                colorized_skeleton[:,:,1] = np.multiply(colorized_skeleton[:,:,1], skeleton_color[1]/255)
                colorized_skeleton[:,:,2] = np.multiply(colorized_skeleton[:,:,2], skeleton_color[2]/255)
                skeletons_image += colorized_skeleton

                # debug_images += [
                    # {'title': f'{feature_type}: contour {i}', 'img': contour_image.copy()},
                    # {'title': f'{feature_type}: skeleton {i}', 'img': skeletonized.copy()},
                    # {'title': f'{feature_type}: colorized skeleton {i}', 'img': colorized_skeleton.copy()},
                # ]
        
        debug_images += [
            {'title': 'skeletons', 'img': skeletons_image.copy()},
        ]

        # parsed_play.type = # need to determine play type
        parsed_play.routes = parsed_routes

        if verbose:
            glog.info(f'..finished parsing {play.title()} in {utils.elapsed_ms(parse_start)}ms: {json.dumps(parsed_play.summary())}')
        if debug:
            utils.display_images(debug_images)
    except Exception as e:
        glog.warning(f'error parsing play: {play.summary()}:\n{traceback.format_exc()}\n{e}\n')
    
    return parsed_play

def parse():
    '''For now access scraped plays via csv and locally downloaded play images'''
    # scraped_playbooks = Playbook.playbooks_from_csv(filepath=constants.PLAYBOOK_CSV_PATH)
    scraped_playbooks = Playbook.read_playbooks_from_json(filepath=constants.SCRAPED_PLAYBOOK_PATH)
    parsed_playbooks = []
    for playbook in scraped_playbooks:
        parse_start = time.time()
        parsed_plays = []
        for scraped_play in playbook.plays:
            pp = _parse_play_image(scraped_play)
            if pp:
                parsed_plays.append(pp)
        parsed_pb = copy.copy(playbook)
        parsed_pb.plays = parsed_plays
        parsed_playbooks.append(parsed_pb)
        glog.info(f'successfully parsed {len(parsed_pb.plays)} / {len(playbook.plays)} plays from {parsed_pb.name} in {utils.elapsed_ms(parse_start)}ms')


    Playbook.write_playbooks_to_json(filepath=constants.PARSED_PLAYBOOK_PATH, playbooks=parsed_playbooks)
