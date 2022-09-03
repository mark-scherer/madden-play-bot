'''
Constants for play animation scraping & parsing.
'''

from os import path

TWITTER_NETLOC = 'twitter.com'

ASSESTS_DIR = path.join(path.dirname(__file__), 'assets')
SCRATCH_DIR = '/tmp/play_animations_scratch/'
PLAY_ANIMATIONS_DIR = '/tmp/play_animations/'
MEDIA_FILENAME = 'animation.mp4'
FRAME_FILENAME = 'frame_0.jpg'

DEBUG_COLOR = [255, 0, 0]
ROUTE_COLORS = [
    [255,0,0],
    [0,255,0],
    [0,0,255],
    [255,255,0],
    [255,0,255],
    [0,255,255],
    [255,255,255],
]

# Fraction of image height marking boundary between field and scoreboard in warped frame.
CROPPED_SCOREBOARD_COEF_WARPED = 0.9

GRASS_COLOR = [45, 195, 96]
GRASS_MASK = {
    'min': [30, 180, 50],
    'max': [120, 200, 120]
}
SIDELINE_MASK = {
    'min': [250, 250, 250],
    'max': [255, 255, 255]
}