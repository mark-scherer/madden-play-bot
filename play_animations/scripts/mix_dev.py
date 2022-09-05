'''Development script for testing mixing colors'''

import sys
from os import path
from typing import Tuple
sys.path.append(path.join(path.dirname(__file__), '..')) # upwards relative imports are hacky
sys.path.append(path.join(path.dirname(__file__), '..', '..')) # upwards relative imports are hacky

import glog

import utils.utils as utils
import utils.color_utils as color_utils
import constants

GRASS_COLOR_RGB = constants.GRASS_COLOR
OFFENSE_COLOR_ADJUSTMENT_HSV = [0, 0, -0.3]
OPACITY = 0.25


# Play animation #1: seahawks @ steelers (offense)
# OFFENSE_COLOR_RGB = [0, 0, 0]
# receiver route
# ROUTE_MIN_HSV = [0.365, 0.65, 0.32]
# ROUTE_MAX_HSV = [0.400, 0.9, 0.38]
# other routes
# ROUTE_MIN_HSV = [0.365, 0.80, 0.50]
# ROUTE_MAX_HSV = [0.400, 0.95, 0.65]

# Play animation #2: rams (offense) @ bengals
# OFFENSE_COLOR_RGB = [0, 52, 153]
# # receiver route
# ROUTE_MIN_HSV = [0.49, 0.8, 0.46]
# ROUTE_MAX_HSV = [0.53, 0.95, 0.54]
# other routes
# ROUTE_MIN_HSV = [0.41, 0.75, 0.55]
# ROUTE_MAX_HSV = [0.46, 0.99, 0.70]

# Play animation #3: bills @ chiefs (offense)
OFFENSE_COLOR_RGB = [229, 22, 47]
# # receiver route
# ROUTE_MIN_HSV = [0.02, 0.45, 0.5]
# ROUTE_MAX_HSV = [0.08, 0.55, 0.65]
# other routes
ROUTE_MIN_HSV = [0.33, 0.45, 0.58]
ROUTE_MAX_HSV = [0.37, 0.60, 0.66]


def _format_tuple(label: str, tuple: Tuple[float]) -> str:
    return f'{label:45}:' \
        f'{tuple[0]:6.2f},' \
        f'{tuple[1]:6.2f},' \
        f'{tuple[2]:6.2f}'

def main():
    route_min_rgb = color_utils.hsv_to_rgb(ROUTE_MIN_HSV)
    route_max_rgb = color_utils.hsv_to_rgb(ROUTE_MAX_HSV)
    
    route_target_hsv = [
        (ROUTE_MIN_HSV[0] + ROUTE_MAX_HSV[0])/2,
        (ROUTE_MIN_HSV[1] + ROUTE_MAX_HSV[1])/2,
        (ROUTE_MIN_HSV[2] + ROUTE_MAX_HSV[2])/2,
    ]
    route_target_rgb = color_utils.hsv_to_rgb(route_target_hsv)

    route_target_range_hsv = [
        ROUTE_MAX_HSV[0] - ROUTE_MIN_HSV[0],
        ROUTE_MAX_HSV[1] - ROUTE_MIN_HSV[1],
        ROUTE_MAX_HSV[2] - ROUTE_MIN_HSV[2],
    ]
    route_target_range_rgb = [
        abs(route_max_rgb[0] - route_min_rgb[0]),
        abs(route_max_rgb[1] - route_min_rgb[1]),
        abs(route_max_rgb[2] - route_min_rgb[2]),
    ]

    offense_color_hsv = color_utils.rgb_to_hsv(OFFENSE_COLOR_RGB)
    altered_offense_color_hsv = [
        utils.clamp(offense_color_hsv[0] + OFFENSE_COLOR_ADJUSTMENT_HSV[0]),
        utils.clamp(offense_color_hsv[1] + OFFENSE_COLOR_ADJUSTMENT_HSV[1]),
        utils.clamp(offense_color_hsv[2] + OFFENSE_COLOR_ADJUSTMENT_HSV[2]),
    ]
    altered_offense_color_rgb = color_utils.hsv_to_rgb(altered_offense_color_hsv)

    mixed_offense_and_grass_rgb = color_utils.overlap_semitransparent_color(
        background_color=GRASS_COLOR_RGB,
        foreground_color=altered_offense_color_rgb,
        foreground_opacity=OPACITY
    )
    mixed_offense_and_grass_hsv = color_utils.rgb_to_hsv(mixed_offense_and_grass_rgb)

    mix_error_hsv = [
        mixed_offense_and_grass_hsv[0] - route_target_hsv[0],
        mixed_offense_and_grass_hsv[1] - route_target_hsv[1],
        mixed_offense_and_grass_hsv[2] - route_target_hsv[2],
    ]


    # Display results
    print()
    print(_format_tuple('route target (HSV)', route_target_hsv))
    print(_format_tuple(f'target range (HSV)', route_target_range_hsv))
    # print(_format_tuple('route target (RGB)', route_target_rgb))
    # print(_format_tuple(f'target range (RGB)', route_target_range_rgb))
    print()
    print(_format_tuple('offense color (HSV)', offense_color_hsv))
    print(_format_tuple(f'darkened offense ({OFFENSE_COLOR_ADJUSTMENT_HSV}) (HSV)', altered_offense_color_hsv))
    print()
    print(_format_tuple(f'mixed ({OPACITY}) (HSV)', mixed_offense_and_grass_hsv))
    print(_format_tuple(f'error (HSV)', mix_error_hsv))
    print()

main()