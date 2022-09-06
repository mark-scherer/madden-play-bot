'''Dev script for finding ball location, field scale in madden play images.'''

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..')) # upwards relative imports are hacky

import cv2
import glog

import constants
import utils.visualization_utils as vis_utils


PLAYBOOK_DIR = '/tmp/madden-play-bot/playbooks/22-213/'
PLAYS_TO_SHOW = 16
BALL_LOCATION_RADIUS = 10



def main():
    play_images_dir = os.path.join(PLAYBOOK_DIR, constants.PLAY_IMAGES_SUBDIR)
    filenames = os.listdir(play_images_dir)
    
    play_images = []
    for filename in filenames[0:PLAYS_TO_SHOW]:
        play_image_filepath = os.path.join(play_images_dir, filename)
        raw_play_image = cv2.imread(play_image_filepath)

        height, width, _ = raw_play_image.shape
        ball_location = (
            int(width * constants.PLAY_IMAGE_BALL_LOCATION_WIDTH_FRAC),
            int(height * constants.PLAY_IMAGE_BALL_LOCATION_HEIGHT_FRAC),
        )
        display_play_image = cv2.circle(
            img=raw_play_image,
            center=ball_location,
            radius=BALL_LOCATION_RADIUS, 
            color=constants.DEBUG_COLOR,
            thickness=-1
        )

        play_images.append({'title': filename, 'img': display_play_image})
        glog.info(f'{filename}: {height} x {width}')


    vis_utils.display_images(play_images)
    

main()