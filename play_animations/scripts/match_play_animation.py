'''Scrape, parse & match play animation.'''

import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '..')) # upwards relative imports are hacky
sys.path.append(path.join(path.dirname(__file__), '..', '..')) # upwards relative imports are hacky

import glog

import play_animation_scraper
import play_animation_parser
from plays import play_matcher
from playbooks.playbook import Playbook

PLAYBOOK_JSON = '/tmp/madden-play-bot/playbooks/22-214/parsed_playbook.json'


# seahawks at steelers (offense)
TEST_PLAY_ANIMATION_URL = 'https://twitter.com/NextGenStats/status/1559573875255431169'

# rams (offense) at bengals
# TEST_PLAY_ANIMATION_URL = 'https://twitter.com/NextGenStats/status/1493714806162878466'

def main():
    # Load playbook
    playbook = Playbook.read_from_json(PLAYBOOK_JSON)
    play_map = {play.id: play for play in playbook.plays}
    glog.info(f'Loaded {len(playbook.plays)} plays from {playbook.title()}')

    # Scrape and parse play animation 
    scraped_play_animation = play_animation_scraper.scrape(url=TEST_PLAY_ANIMATION_URL)
    parsed_play = play_animation_parser.parse(scraped_play_animation=scraped_play_animation)
    glog.info(f'Parsed play animation from {TEST_PLAY_ANIMATION_URL}')

    # Find best match in playbook
    match_result = play_matcher.match_play(play=parsed_play, possible_matches=playbook.plays)
    matched_plays = [(play_map[match.truth_play_id], match.score) for match in match_result]
    matched_plays = sorted(matched_plays, key=lambda match: match[1])

    # glog.info(f'Got matches for {parsed_play.title()}: {[f"{match[0].title()}: {match[1]}" for match in matched_plays[0:3]]}')

main()