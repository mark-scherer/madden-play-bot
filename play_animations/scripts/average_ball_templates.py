'''
Smoothly combine multiple individual ball templates.
'''

import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '..')) # upwards relative imports are hacky
sys.path.append(path.join(path.dirname(__file__), '..', '..', 'utils')) # upwards relative imports are hacky

import glog
import cv2
import numpy as np

import utils
import constants

BALL_TEMPLATE_FILENAMES = [
    'ball_bills_chiefs.png',
    'ball_rams_bengals_0.png',
    'ball_rams_bengals_1.png',
]

TEMPLATE_COMMON_SIZE = (12, 16)  # w x h
BLUR_KERNEL_SIZE = (3, 3)  # must be odd
TEMPLATE_CROP_FRACS = [0.09, 0, 0.92, 1]

OUTPUT_FILENAME = 'ball_combined.png'

def main():
    input_templates = []
    debug_images = []
    for i, filename in enumerate(BALL_TEMPLATE_FILENAMES):
        input_path = path.join(constants.ASSESTS_DIR, filename)
        template = cv2.imread(input_path)
        display_template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
        
        input_templates.append(display_template)
        # debug_images.append({'title': f'input template: {i}', 'img': display_template})

    resized_templates = [cv2.resize(template, TEMPLATE_COMMON_SIZE) for template in input_templates]
    debug_images += [{'title': f'resized template: {i}', 'img': resized} for i, resized in enumerate(resized_templates)]

    avg_template = np.zeros(resized_templates[0].shape, np.float32)
    for template in resized_templates:
        avg_template = cv2.accumulate(template, avg_template)
    avg_template = (avg_template / len(resized_templates)).astype(np.uint8)
    debug_images.append({'title': 'avg template', 'img': avg_template})

    blurred_avg_template = cv2.GaussianBlur(avg_template, BLUR_KERNEL_SIZE, 0)
    debug_images.append({'title': 'blurred template', 'img': blurred_avg_template})

    cropped_template, _ = utils.crop_image(input=blurred_avg_template, crop_fracs=TEMPLATE_CROP_FRACS)
    debug_images.append({'title': 'cropped template', 'img': cropped_template})

    output_path = path.join(constants.ASSESTS_DIR, OUTPUT_FILENAME)
    writeable_template = cv2.cvtColor(cropped_template, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, writeable_template)
    glog.info(f'saved template image to {output_path}')
    utils.display_images(images=debug_images)

main()