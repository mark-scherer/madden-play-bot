'''Find best match for given play.'''

import sys
from os import path
from typing import List, NamedTuple, Tuple
import copy
from functools import lru_cache
sys.path.append(path.join(path.dirname(__file__))) # relative imports are weird
sys.path.append(path.join(path.dirname(__file__), '..')) # upwards relative imports are hacky

import glog
import numpy as np
import cv2

import constants
from play import Play, PlayMask
import visualization_utils as vis_utils
import image_utils

# Radius in yards to blur ruth routes in all directions before comparing plays.
ROUTE_BLURRING_DISTANCE_YARDS = 7


class PlayMatch(NamedTuple):
    '''Matching score for single play combination.'''
    test_play_id: int
    truth_play_id: int
    score: float
    debug_images: List[np.array]


def _blur_play(play: Play) -> Play:
    '''Blur Play's PlayMask in preparation for matching comparisons.
    
    Return: Copy of Play with PlayMask blurred.
    '''
    blur_kernel_size_component = \
        2*int(ROUTE_BLURRING_DISTANCE_YARDS * constants.PLAYMASK_SCALE) + 1
    blur_kernel_size = (blur_kernel_size_component, blur_kernel_size_component)
    blurred_mask = cv2.GaussianBlur(
        src=play.playmask.mask,
        ksize=blur_kernel_size,
        sigmaX=0
    )

    mask_dir, raw_mask_filename = path.split(play.playmask.mask_local_path)
    mask_filename, mask_ext = path.splitext(raw_mask_filename)
    result_local_path = f'{mask_dir}/{mask_filename}_blurred{mask_ext}'
    PlayMask.save_mask(mask=blurred_mask, filepath=result_local_path)

    result = copy.deepcopy(play)
    result.playmask.mask_local_path = result_local_path
    result.playmask.mask = blurred_mask
    
    return result


# @lru_cache(lambda mask: mask.mask_local_path)
# def _sum_playmask(mask: np.array) -> int:
#     '''Helper to enabling caching of mask sums.'''
#     return cv2.sum(mask)


def _play_overlap(test_play: Play, truth_play: Play) -> Tuple[float, List[np.array]]:
    '''Compute average value of test_play at truth_play's mask pixels.

    Return:
        1. calculated match score
        2. list of debug images
    '''
    debug_images = []

    test_playmask = test_play.playmask.mask
    truth_playmask = truth_play.playmask.mask

    # Filter blurred test_play to truth_play's pixels
    glog.info(f'attempting to filer test playmask ({test_playmask.dtype}, {test_playmask.shape}) by truth playmask ({truth_playmask.dtype}, {truth_playmask.shape})')
    filtered_test_playmask = cv2.bitwise_and(test_playmask, test_playmask, mask=truth_playmask)
    # debug_images.append({'title': 'filtered test playmask', 'img': filtered_test_playmask})

    # Calculate overlap score
    # filtered_test_sum = cv2.sum(filtered_test_playmask)
    # test_sum = cv2.sum(test_playmask)

    filtered_test_sum = np.sum(np.concatenate(filtered_test_playmask))
    test_sum = np.sum(np.concatenate(test_playmask))

    overlap_score = filtered_test_sum / test_sum
    glog.info(f'got match score for {test_play.title()} & {truth_play.title()}: {round(overlap_score, 3)} ({filtered_test_sum} / {test_sum})')

    # Generate overlap image
    overlap_image = cv2.cvtColor(test_playmask.copy(), cv2.COLOR_GRAY2RGB)
    max_mask_value = np.amax(test_playmask)
    overlap_image *= int(255 / max_mask_value)

    truth_play_pixels = image_utils.get_mask_white_pixels(truth_playmask)
    for pixel in truth_play_pixels:
        overlap_image[pixel[1], pixel[0], 0] = 255
        overlap_image[pixel[1], pixel[0], 1] = 0
        overlap_image[pixel[1], pixel[0], 2] = 0
    debug_images.append({'title': f'{truth_play.title()}: {round(overlap_score, 3)}', 'img': overlap_image})

    return (overlap_score, debug_images)


def match_play(play: Play, possible_matches: List[Play]) -> List[PlayMatch]:
    '''Determine match score for play against all possible_matches.'''
    debug_images = []

    assert len(possible_matches) > 0, 'no possible_plays provided.'

    debug_images.append({
        'title': 'test play mask',
        'img': play.playmask.mask
    })
    
    # Resize test play to truth_play size
    truth_play_height, truth_play_width = possible_matches[0].playmask.mask.shape
    resized_truth_play = copy.deepcopy(play)
    resized_truth_play.playmask = PlayMask.resize(
        input_playmask=resized_truth_play.playmask,
        new_width=truth_play_width,
        new_height=truth_play_height
    )

    # Blur test play for fuzzier matching
    blurred_test_play = _blur_play(play=resized_truth_play)
    debug_images += [{'title': 'blurred test play', 'img': blurred_test_play.playmask.mask}]
    glog.info(f'blurred: {play.playmask.mask.shape} -> {blurred_test_play.playmask.mask.shape}')

    play_match_results = []
    for truth_play in possible_matches:
        match_score, match_debug_images = _play_overlap(test_play=blurred_test_play, truth_play=truth_play)
        play_match_results.append(PlayMatch(
            test_play_id=play.id,
            truth_play_id=truth_play.id,
            score=match_score,
            debug_images=match_debug_images
        ))

    # Add match debug images in order
    sorted_match_results = sorted(play_match_results, key=lambda match_result: match_result.score, reverse=True)
    for i, match_result in enumerate(sorted_match_results):
        debug_images += match_result.debug_images
        glog.info(f'match_result {i}: {round(match_result.score, 3)}: {match_result.truth_play_id}')
    
    vis_utils.display_images(debug_images)

    # return [PlayMatch(
    #     test_play_id=play.id,
    #     truth_play_id=other_play.id,
    #     score=0
    # ) for other_play in possible_matches]