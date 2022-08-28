'''Helper script for finding exact color bounds of routes in
different play animations.
'''

import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '..')) # upwards relative imports are hacky
sys.path.append(path.join(path.dirname(__file__), '..', '..', 'utils')) # upwards relative imports are hacky

import numpy as np
import cv2 
import glog

import utils
import constants
import play_animation_scraper
import play_animation_parser
import play_animation_preprocessor as preprocessor
import play_animation_parser as parser


# Play animation #1: seahawks @ steelers (offense)
# Offense color:
    # RGB : 0, 0, 0
    # HSV: 0.0, 0.0, 0.0
# URL = 'https://twitter.com/NextGenStats/status/1559573875255431169'
# Finalized bounds for reciever route pre-catch
# ROUTE_HSV_MIN = [0.365, 0.65, 0.32]
# ROUTE_HSV_MAX = [0.400, 0.9, 0.38]
# Finalized bounds for non-reciever routes
# ROUTE_HSV_MIN = [0.365, 0.80, 0.50]
# ROUTE_HSV_MAX = [0.400, 0.95, 0.65]

# Play animation #2: rams (offense) @ bengals
# Offense color:
    # RGB : 0, 52, 153
    # HSV: 0.61, 1.0, 0.60
# URL = 'https://twitter.com/NextGenStats/status/1493714806162878466'
# Finalized bounds for reciever route pre-catch
# ROUTE_HSV_MIN = [0.49, 0.8, 0.46]
# ROUTE_HSV_MAX = [0.53, 0.95, 0.54]
# Finalized bounds for non-reciever routes
# ROUTE_HSV_MIN = [0.41, 0.75, 0.55]
# ROUTE_HSV_MAX = [0.46, 0.99, 0.70]

# Play animation #3: bills @ chiefs (offense)
# Offense color:
    # RGB : 229, 22, 47
    # HSV: 0.98, 0.90, 0.90
URL = 'https://twitter.com/NextGenStats/status/1485447699600003077'
# Finalized bounds for reciever route pre-catch
# ROUTE_HSV_MIN = [0.02, 0.45, 0.5]
# ROUTE_HSV_MAX = [0.08, 0.55, 0.65]
# Finalized bounds for non-reciever routes
ROUTE_HSV_MIN = [0.33, 0.45, 0.58]
ROUTE_HSV_MAX = [0.37, 0.60, 0.66]


def load_and_preprocess(media_url: str) -> np.array:
    scraped_play_animation = play_animation_scraper.scrape(url=media_url)
    input_frame_path = path.join(scraped_play_animation.media_dir, constants.FRAME_FILENAME)
    input_frame = cv2.imread(input_frame_path)
    display_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

    preprocessed_frame, _, ball_location, preprocess_debug_images = preprocessor.preprocess_frame(display_frame=display_frame)
    return display_frame, preprocessed_frame, ball_location

def main():
    '''Given one play animation help find exact color bounds of routes.'''
    debug_images = []

    # Load animation
    display_frame, preprocessed_frame, ball_location = load_and_preprocess(media_url=URL)

    # Get offense color
    offense_color, offense_color_debug_images = parser._get_offense_color(
        display_frame=display_frame,
        preprocessed_frame=preprocessed_frame,
        ball_location=ball_location)
    glog.info(f'got offense color: RGB: {offense_color}, HSV: {utils.rgb_to_hsv(offense_color)}')

    # Get HSV frame
    full_size_frame = preprocessed_frame.astype(np.float32)/255.
    hsv_frame = cv2.cvtColor(full_size_frame, cv2.COLOR_RGB2HSV)
    hsv_frame[:,:,0] /= 360.
    debug_images.append({'title': 'HSV frame', 'img': hsv_frame})

    # Mask routes
    route_mask = utils.img_threshold_by_range(
        img=hsv_frame,
        min=ROUTE_HSV_MIN,
        max=ROUTE_HSV_MAX,
        reverse=False
    )
    debug_images.append({'title': 'Route mask', 'img': route_mask})

    utils.display_images(images=debug_images)

main()