'''Dev script for testing play_animation / playbook play overlap algo.

To be used on known play matches.
'''

import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '..')) # upwards relative imports are hacky
sys.path.append(path.join(path.dirname(__file__), '..', '..')) # upwards relative imports are hacky

import glog
import cv2

import play_animation_scraper
import play_animation_parser
from plays import play_matcher
from playbooks.playbook import Playbook
import visualization_utils as vis_utils


# seahawks at steelers (offense)
# GUN SPREAD Y SLOT WK (4669) - Four Verticals
    # this is a near perfect match objectively
TEST_PLAY_ANIMATION_URL = 'https://twitter.com/NextGenStats/status/1559573875255431169'
PLAYBOOK_JSON = '/tmp/madden-play-bot/playbooks/22-214/parsed_playbook.json'
PLAY_ID = 71651

# rams at bengals (offense)
# GUN TREY OPEN (8409) - Salem
    # kinda, no great match in bengals playbook
# TEST_PLAY_ANIMATION_URL = 'https://twitter.com/NextGenStats/status/1493714866200199172'
# PLAYBOOK_JSON = '/tmp/madden-play-bot/playbooks/22-150/parsed_playbook.json'
# PLAY_ID = 50670

# bill at chiefs (offense)
# cannot find matching play

# rams (offense) at bengals
    # no real match...



def main():
    debug_images = []

    # Load target play/
    playbook = Playbook.read_from_json(PLAYBOOK_JSON)
    play_map = {play.id: play for play in playbook.plays}
    truth_play = play_map[PLAY_ID]
    glog.info(f'Loaded {len(playbook.plays)} plays from {playbook.title()} - targeting {truth_play.title()}')

    play_image = cv2.imread(truth_play.image_local_path)
    debug_images.append({'title': 'madden play', 'img': play_image})

    # Scrape and parse play animation 
    scraped_play_animation = play_animation_scraper.scrape(url=TEST_PLAY_ANIMATION_URL)
    parsed_play = play_animation_parser.parse(scraped_play_animation=scraped_play_animation)
    glog.info(f'Parsed play animation from {TEST_PLAY_ANIMATION_URL}')

    # Compute match
    match_result, match_debug_images = play_matcher.match_play(play=parsed_play, possible_matches=[truth_play])
    debug_images += match_debug_images

    vis_utils.display_images(debug_images)


main()