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
from playbook import Play, Formation, Route, Point

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

RECEIVER_ROUTE_COLOR_OPACITY = 0.55  # opacity when mixing offense color over grass for target color
RECEIVER_ROUTE_MASK_THRESHOLD = [
    0.04,
    0.15,
    0.08 
]  # HSV, +/- margin around route color when masking routes
OTHER_ROUTES_COLOR_OPACITY = 0.25
OTHER_ROUTES_MIXED_HSV_ADJUSTMENT = [
    0,
    0.08,
    0
]
OTHER_ROUTES_MASK_THRESHOLD = [
    0.08,
    0.14,
    0.08 
]

OVERLAP_CALC_DILATION_KERNEL_SIZE = (3, 3)
OVERLAP_REMOVAL_DILATION_KERNEL_SIZE = (3, 3)

RECEIVER_ROUTE_CLEANING_KERNEL_SIZE = (3, 3)
OTHER_ROUTES_CLEANING_KERNEL_SIZE = (2, 2)
RECEIVER_ROUTE_CLOSING_KERNEL_SIZE = (10, 10)
OTHER_ROUTES_CLOSING_KERNEL_SIZE = (10, 10)


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
    ball_location: Point
) -> Tuple[Tuple[int], List[np.array]]:
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


def _get_route_masks(
    preprocessed_frame: np.array,
    offense_color: Tuple[int, int, int]
) -> Tuple[List[np.array], List[np.array]]:
    '''Return mask of routes given processed frame and offense color.
    
    Return:
    1. list of uint8 binary mask of parsed routes
    2. list of debug images

    Notes:
    1. HSV seems to work as well as HSL
    '''
    debug_images = []

    # First calculate target route colors with MOE
    receiver_route_color = utils.overlap_semitransparent_color(
        background_color=constants.GRASS_COLOR,
        foreground_color=offense_color,
        foreground_opacity=RECEIVER_ROUTE_COLOR_OPACITY
    )
    receiver_route_color_hsv = utils.rgb_to_hsv(receiver_route_color)

    receiver_route_mask_min = [
        utils.clamp(receiver_route_color_hsv[0] - RECEIVER_ROUTE_MASK_THRESHOLD[0]),
        utils.clamp(receiver_route_color_hsv[1] - RECEIVER_ROUTE_MASK_THRESHOLD[1]),
        utils.clamp(receiver_route_color_hsv[2] - RECEIVER_ROUTE_MASK_THRESHOLD[2]),
    ]
    receiver_route_mask_max = [
        utils.clamp(receiver_route_color_hsv[0] + RECEIVER_ROUTE_MASK_THRESHOLD[0]),
        utils.clamp(receiver_route_color_hsv[1] + RECEIVER_ROUTE_MASK_THRESHOLD[1]),
        utils.clamp(receiver_route_color_hsv[2] + RECEIVER_ROUTE_MASK_THRESHOLD[2]),
    ]

    other_routes_color = utils.overlap_semitransparent_color(
        background_color=constants.GRASS_COLOR,
        foreground_color=offense_color,
        foreground_opacity=OTHER_ROUTES_COLOR_OPACITY
    )
    other_routes_color_hsv = utils.rgb_to_hsv(other_routes_color)
    other_routes_color_hsv = [
        utils.clamp(other_routes_color_hsv[0] + OTHER_ROUTES_MIXED_HSV_ADJUSTMENT[0]),
        utils.clamp(other_routes_color_hsv[1] + OTHER_ROUTES_MIXED_HSV_ADJUSTMENT[1]),
        utils.clamp(other_routes_color_hsv[2] + OTHER_ROUTES_MIXED_HSV_ADJUSTMENT[2]),
    ]

    other_routes_mask_min = [
        utils.clamp(other_routes_color_hsv[0] - OTHER_ROUTES_MASK_THRESHOLD[0]),
        utils.clamp(other_routes_color_hsv[1] - OTHER_ROUTES_MASK_THRESHOLD[1]),
        utils.clamp(other_routes_color_hsv[2] - OTHER_ROUTES_MASK_THRESHOLD[2]),
    ]
    other_routes_mask_max = [
        utils.clamp(other_routes_color_hsv[0] + OTHER_ROUTES_MASK_THRESHOLD[0]),
        utils.clamp(other_routes_color_hsv[1] + OTHER_ROUTES_MASK_THRESHOLD[1]),
        utils.clamp(other_routes_color_hsv[2] + OTHER_ROUTES_MASK_THRESHOLD[2]),
    ]

    # Get raw route masks from HSV frame
    full_size_normalized_frame = preprocessed_frame.astype(np.float32)/255.
    hsv_frame = cv2.cvtColor(full_size_normalized_frame, cv2.COLOR_RGB2HSV)
    hsv_frame[:,:,0] /= 360.
    # debug_images.append({'title': 'HSV frame', 'img': hsv_frame})

    receiver_route_mask = utils.img_threshold_by_range(
        img=hsv_frame,
        min=receiver_route_mask_min,
        max=receiver_route_mask_max,
        reverse=False
    )
    # debug_images.append({'title': 'receiver route mask', 'img': receiver_route_mask})
    
    other_routes_mask = utils.img_threshold_by_range(
        img=hsv_frame,
        min=other_routes_mask_min,
        max=other_routes_mask_max,
        reverse=False
    )
    # debug_images.append({'title': 'other routes mask', 'img': other_routes_mask})

    # Clean route masks: start by removing any overlap between two reciever masks.
        # Because masks target different colors, overlap likely to be junk.
    overlap_calc_dilation_kernel = np.ones(OVERLAP_CALC_DILATION_KERNEL_SIZE, np.uint8)
    diluted_receiver_route_mask = cv2.morphologyEx(receiver_route_mask, cv2.MORPH_DILATE, overlap_calc_dilation_kernel)
    diluted_other_routes_mask = cv2.morphologyEx(other_routes_mask, cv2.MORPH_DILATE, overlap_calc_dilation_kernel)

    mask_overlap = cv2.bitwise_and(diluted_receiver_route_mask, diluted_other_routes_mask)
    # debug_images.append({'title': 'masks overlap', 'img': mask_overlap})
    
    overlap_removal_dilation_kernel = np.ones(OVERLAP_REMOVAL_DILATION_KERNEL_SIZE, np.uint8)
    diluted_mask_overlap = cv2.morphologyEx(mask_overlap, cv2.MORPH_DILATE, overlap_removal_dilation_kernel)
    # debug_images.append({'title': 'diluted masks overlap', 'img': diluted_mask_overlap})

    # Only use diluted overlap for other routes mask - for reciever route mask diluation removes the actual route.
    receiver_route_mask_minus_overlap = cv2.bitwise_and(receiver_route_mask, receiver_route_mask, mask=cv2.bitwise_not(mask_overlap))
    other_routes_mask_minus_overlap = cv2.bitwise_and(other_routes_mask, other_routes_mask, mask=cv2.bitwise_not(diluted_mask_overlap))
    # debug_images.append({'title': 'receiver routes mask - overlap', 'img': receiver_route_mask_minus_overlap})
    # debug_images.append({'title': 'other routes mask - overlap', 'img': other_routes_mask_minus_overlap})

    # Next perform standard mask cleaning.
    reciever_route_cleaning_kernel = np.ones(RECEIVER_ROUTE_CLEANING_KERNEL_SIZE, np.uint8)
    receiver_route_cleaned_mask = cv2.morphologyEx(receiver_route_mask_minus_overlap, cv2.MORPH_OPEN, reciever_route_cleaning_kernel)
    other_routes_cleaning_kernel = np.ones(OTHER_ROUTES_CLEANING_KERNEL_SIZE, np.uint8)
    other_routes_cleaned_mask = cv2.morphologyEx(other_routes_mask_minus_overlap, cv2.MORPH_OPEN, other_routes_cleaning_kernel)
    # debug_images.append({'title': 'cleaned receiver route mask', 'img': receiver_route_cleaned_mask})
    # debug_images.append({'title': 'cleaned other routes mask', 'img': other_routes_cleaned_mask})

    # Try to close masks to connect separate parts
    reciever_route_closing_kernel = np.ones(RECEIVER_ROUTE_CLOSING_KERNEL_SIZE, np.uint8)
    receiver_route_closed_mask = cv2.morphologyEx(receiver_route_cleaned_mask, cv2.MORPH_CLOSE, reciever_route_closing_kernel)
    other_routes_closing_kernel = np.ones(OTHER_ROUTES_CLOSING_KERNEL_SIZE, np.uint8)
    other_routes_closed_mask = cv2.morphologyEx(other_routes_cleaned_mask, cv2.MORPH_CLOSE, other_routes_closing_kernel)
    debug_images.append({'title': 'closed receiver route mask', 'img': receiver_route_closed_mask})
    debug_images.append({'title': 'closed other routes mask', 'img': other_routes_closed_mask})

    return (
        [receiver_route_closed_mask, other_routes_closed_mask], 
        debug_images
    )


def scale_routes(raw_routes: List[Route], ball_location: Point, field_scale: float) -> List[Route]:
    '''Convert routes from pixel coordinates to route_coordinates.'''
    scaled_routes = []
    for raw_route in raw_routes:
        scaled_points = []
        for raw_point in raw_route.points:
            # convert to ball-centric coordinates
            x = raw_point.x - ball_location.x
            y = raw_point.y - ball_location.y

            # scale to yards
            x /= field_scale
            y /= field_scale

            # flip y (in pixel coordinates, y=0 at top of screen)
            y *= -1

            scaled_points.append(Point(x=x, y=y))
        scaled_routes.append(Route(
            points=scaled_points
        ))
    return scaled_routes


def parse(scraped_play_animation: PlayAnimation) -> PlayAnimation:
    '''Parse play out of a scraped play animation'''
    debug_images = []

    input_frame_path = path.join(scraped_play_animation.media_dir, constants.FRAME_FILENAME)
    input_frame = cv2.imread(input_frame_path)
    display_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB) 
    # debug_images = [{
    #     'title': 'input frame',
    #     'img': display_frame
    # }]
    
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
    # glog.info(f'offensive color: {offense_color} (hsv: {utils.rgb_to_hsv(offense_color)})')
    glog.info(f'offensive color: {offense_color} (hsl: {utils.rgb_to_hsl(offense_color)})')

    routes_masks, route_masks_debug_images = _get_route_masks(
        preprocessed_frame=preprocessed_frame,
        offense_color=offense_color,
    )
    debug_images += route_masks_debug_images

    routes, routes_debug_images = utils.parse_routes_from_masks(routes_masks)
    glog.info(f'parsed {len(routes)} routes')
    debug_images += routes_debug_images

    scaled_routes = scale_routes(raw_routes=routes, ball_location=ball_location, field_scale=field_scale)
    parsed_play = Play(
        id=-1,
        name='unknown',
        formation=Formation(
            id=-1,
            name='unknown',
            family=-1
        ),
        routes=scaled_routes
    )

    # utils.display_images(images=debug_images)
    utils.show_play(parsed_play)
    return parsed_play