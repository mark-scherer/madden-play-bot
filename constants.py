'''
Constants for play animation scraping & parsing.
'''

from os import path

ASSESTS_DIR = path.join(path.dirname(__file__), 'play_animations', 'assets')

BASE_DIR = '/tmp/madden-play-bot/'
PLAY_ANIMATIONS_DIR = path.join(BASE_DIR, 'play_animations')
SCRATCH_DIR = path.join(BASE_DIR, 'scratch')
PLAYBOOKS_BASE_DIR = path.join(BASE_DIR, 'playbooks')
PLAY_IMAGES_SUBDIR = 'images'

PLAYMASK_FILENAME = 'playmask.png'
SCRAPED_PLAYBOOK_DATA_FILENAME = 'scraped_playbook.json'
PARSED_PLAYBOOK_DATA_FILENAME = 'parsed_playbook.json'

MEDIA_FILENAME = 'animation.mp4'
FRAME_FILENAME = 'frame_0.jpg'

DEBUG_COLOR = [255, 0, 0]

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

FIELD_WIDTH_YARDS = 160/3

PLAY_IMAGE_BALL_LOCATION_WIDTH_FRAC = 0.5
PLAY_IMAGE_BALL_LOCATION_HEIGHT_FRAC = 0.4

PLAYMASK_SCALE = 3  # pixels / yard in parsed play data