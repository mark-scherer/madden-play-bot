'''
Functions for parsing already scraped play images into usable data.
'''

from typing import List

import glog
import csv

from constants import PLAYBOOK_CSV_PATH
from playbook import Playbook, Play

def _parse_play_image(play: Play) -> Play:
    '''Parse play image into usable data and return as new Play object'''
    parsed = None
    try:
        raise NotImplementedError()
    except Exception as e:
        glog.warning(f'error parsing play: {play.summary()}: {e}')
    
    return parsed

def parse():
    '''For now access scraped plays via csv and locally downloaded play images'''
    playbooks = Playbook.playbooks_from_csv(filepath=PLAYBOOK_CSV_PATH)
    all_plays = []
    for playbook in playbooks:
        all_plays += playbook.plays
    
    parsed_plays = []
    for play in all_plays:
        pp = _parse_play_image(play)
        if pp:
            parsed_plays.append(pp)

    glog.info(f'..successfully parsed {len(parsed_plays)} / {len(all_plays)} plays.')
