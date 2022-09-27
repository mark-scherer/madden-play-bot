'''
Functions for parsing already scraped play images into usable data.
'''

import sys
from os import path
from typing import List, Dict, Any, Tuple, NamedTuple
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
from plays.play import Play, PlayMask, Point, Formation, SpacingCorrection
from playbooks.playbook import Playbook

FORMATION_THRESHOLD = {
    'min': [210, 210, 210],
    'max': [255, 255, 255],
}
MADDEN_MIN_PLAYER_SPACING = 32  # px, in original images
FORMATION_CORE_CROP_HEIGHT = 1.5 * MADDEN_MIN_PLAYER_SPACING  # px behind LOS to look for adjacent players in core
FORMATION_CORE_UPSCALE_FACTOR = 2.5
FORMATION_BACKFIELD_UPSCALE_FACTOR = 2

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


def parse_formation(
    input_formation: Formation,
    play_image: np.array,
    formation_mask_path: str,
    ball_location: Point
) -> Tuple[Formation, List[np.array]]:
    '''Parses formation and returns modified copy.'''
    debug_images = []
    parsed_formation = copy.deepcopy(input_formation)

    # Parse and save formation mask.
    formation_mask = image_utils.img_threshold_by_range(
        img=play_image,
        min=FORMATION_THRESHOLD['min'],
        max=FORMATION_THRESHOLD['max'],
    )
    debug_images.append({'title': 'formation_mask', 'img': formation_mask})

    PlayMask.save_mask(mask=formation_mask, filepath=formation_mask_path)
    parsed_formation.mask = PlayMask(
        ball_location=ball_location,
        mask_local_path=formation_mask_path,
        mask=formation_mask
    )

    # Calculate horizontal spacing correction.
    spacing_correction, correction_debug_images = _get_formation_spacing_correction(
        formation=parsed_formation,
        play_image=play_image
    )
    parsed_formation.spacing_correction = spacing_correction
    debug_images += correction_debug_images

    # Apply horizontal spacing correction.
    respaced_formation_mask = PlayMask.apply_horizontal_spacing_correction(
        input_playmask=parsed_formation.mask,
        correction=spacing_correction
    )
    debug_images.append({'title': 'respaced formation_mask', 'img': respaced_formation_mask.mask})

    # Apply backfield vertical scaling correction.
    rescaled_backfield_formation_mask = PlayMask.apply_backfield_vertical_scaling(
        input_playmask=respaced_formation_mask,
        backfield_scaling_factor=1/FORMATION_BACKFIELD_UPSCALE_FACTOR
    )
    debug_images.append({'title': 'rescaled backfield formation_mask', 'img': rescaled_backfield_formation_mask.mask})

    parsed_formation.mask = rescaled_backfield_formation_mask
    return (parsed_formation, debug_images)


def _get_formation_spacing_correction(formation: Formation, play_image: np.array) -> Tuple[SpacingCorrection, List[np.array]]:
    '''Calculate horizontal spacing correction for a madden play.
    Madden upscales formation core causing inconsistent spacing with space outside the TEs.
    If this isn't corrected WR routes are places too far outside and don't match actual plays.
    
    TODO: implement caching
    '''
    debug_images = []
    formation_playmask = formation.mask
    formation_mask = formation_playmask.mask
    ball_x = formation_playmask.ball_location.x
    los_y = formation_playmask.ball_location.y
    
    next_player_crop_ymin = int(los_y)
    next_player_crop_ymax = int(los_y + FORMATION_CORE_CROP_HEIGHT)

    # Find the width of the core of the formation by iterating outwards from the ball.
    left_edge_x = ball_x - 0.5*MADDEN_MIN_PLAYER_SPACING
    steps_left = 0
    found_widest_left = False
    while not found_widest_left:
        steps_left += 1
        next_player_xmin = int(left_edge_x - MADDEN_MIN_PLAYER_SPACING)
        next_player_xmax = int(left_edge_x)
        next_player_crop = formation_mask[
            next_player_crop_ymin:next_player_crop_ymax,
            next_player_xmin:next_player_xmax
        ]

        # debug_images.append({'title': f'left player crop: {steps_left}', 'img': next_player_crop})

        next_player_pixels = image_utils.get_mask_white_pixels(next_player_crop)
        if len(next_player_pixels) > 0:
            #  note: purposely decoupling the step left and the next play crop xmin - could be different
            left_edge_x -= MADDEN_MIN_PLAYER_SPACING
        else:
            found_widest_left = True

    right_edge_x = ball_x + 0.5*MADDEN_MIN_PLAYER_SPACING
    steps_right = 0
    found_widest_right = False
    while not found_widest_right:
        steps_right += 1
        next_player_xmin = int(right_edge_x)
        next_player_xmax = int(right_edge_x + MADDEN_MIN_PLAYER_SPACING)
        next_player_crop = formation_mask[
            next_player_crop_ymin:next_player_crop_ymax,
            next_player_xmin:next_player_xmax
        ]

        # debug_images.append({'title': f'right player crop: {steps_right}', 'img': next_player_crop})

        next_player_pixels = image_utils.get_mask_white_pixels(next_player_crop)
        if len(next_player_pixels) > 0:
            #  note: purposely decoupling the step left and the next play crop xmin - could be different
            right_edge_x += MADDEN_MIN_PLAYER_SPACING
        else:
            found_widest_right = True

    glog.info(f'found edges of formation core: left: {left_edge_x} ({steps_left} steps), right: {right_edge_x} ({steps_right} steps)')

    core_width_left = ball_x - left_edge_x
    core_width_right = right_edge_x - ball_x
    corrected_core_width_left = core_width_left / FORMATION_CORE_UPSCALE_FACTOR
    corrected_core_width_right = core_width_right / FORMATION_CORE_UPSCALE_FACTOR

    correction = SpacingCorrection(
        original_xmin=left_edge_x,
        original_xmax=right_edge_x,
        corrected_xmin=ball_x-corrected_core_width_left,
        corrected_xmax=ball_x+corrected_core_width_right
    )

    return (correction, debug_images)


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

        # Find ball.
        ball_location = Point(
            x=width * constants.PLAY_IMAGE_BALL_LOCATION_WIDTH_FRAC,
            y=width * constants.PLAY_IMAGE_BALL_LOCATION_HEIGHT_FRAC,
        )

        # Parse formation.
        formation_mask_path = path.join(constants.FORMATIONS_DIR, f'{play.formation.id}_{constants.PLAYMASK_FILENAME}')
        parsed_formation, formation_debug_images = parse_formation(
            input_formation=play.formation,
            play_image=img,
            formation_mask_path=formation_mask_path,
            ball_location=ball_location
        )
        debug_images += formation_debug_images

        # Find masks for each route type.
        masks = {}
        for feature_type, mask_thresholds in MASK_THRESHOLDS.items():
            mask = image_utils.img_threshold_by_range(img, min=mask_thresholds['min'], max=mask_thresholds['max'])
            closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSING_KERNERL)
            masks[feature_type] = closed_mask

        # Generate PlayMask.
        play_dir = path.dirname(play.image_local_path)
        play_filename = path.splitext(path.basename(play.image_local_path))[0]
        playmask_path = path.join(play_dir, f'{play_filename}_{constants.PLAYMASK_FILENAME}')
        parsed_playmask, playmask_debug_images = image_utils.parse_playmask_from_masks(
            masks=masks.values(),
            ball_location=ball_location,
            playmask_path=playmask_path
        )
        debug_images += playmask_debug_images

        # Account for inconsistent horizontal spacing.
        respaced_playmask = PlayMask.apply_horizontal_spacing_correction(
            input_playmask=parsed_playmask,
            correction=parsed_formation.spacing_correction
        )
        # debug_images.append({'title': 'respaced playmask', 'img': respaced_playmask.mask})

        # Account for inconsistent backfield vertical scaling.
        rescaled_backfield_playmask = PlayMask.apply_backfield_vertical_scaling(
            input_playmask=respaced_playmask,
            backfield_scaling_factor=1/FORMATION_BACKFIELD_UPSCALE_FACTOR
        )
        debug_images.append({'title': 'rescaled backfield playmask', 'img': rescaled_backfield_playmask.mask})

        # Downsample Playmask.
        sampled_playmask = PlayMask.resample(rescaled_backfield_playmask, constants.PLAYMASK_SCALE)
        # debug_images.append({'title': 'sampled playmask', 'img': sampled_playmask.mask})

        # Standardize vertical cropping.
        cropped_playmask = PlayMask.crop_field_vertically(input_playmask=sampled_playmask)
        # debug_images.append({'title': 'cropped playmask', 'img': cropped_playmask.mask})

        # parsed_play.type = # need to determine play type
        parsed_play.formation = parsed_formation
        parsed_play.playmask = cropped_playmask

        glog.info(f'..finished parsing {play.title()} in {utils.elapsed_ms(parse_start)}ms: {json.dumps(parsed_play.summary())}')
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
    for scraped_play in scraped_playbook.plays[6:7]:
        parsed_play = _parse_play_image(scraped_play)
        if parsed_play:
            parsed_playbook.plays.append(parsed_play)

    # DEBUG: play scaling issues
    vis_utils.show_plays(parsed_playbook.plays)

    glog.info(f'successfully parsed {len(parsed_playbook.plays)} / {len(scraped_playbook.plays)} plays from {parsed_playbook.title()} in {utils.elapsed_ms(parse_start)}ms')
    parsed_playbook_filepath = path.join(playbook_dir, constants.PARSED_PLAYBOOK_DATA_FILENAME)
    parsed_playbook.write_to_json(filepath=parsed_playbook_filepath)
