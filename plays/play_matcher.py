'''Find best match for given play.'''

import sys
from os import path
from typing import List, NamedTuple
sys.path.append(path.join(path.dirname(__file__))) # relative imports are weird
sys.path.append(path.join(path.dirname(__file__), '..')) # upwards relative imports are hacky

import glog

from play import Play
import visualization_utils as vis_utils


class PlayMatch(NamedTuple):
    '''Matching score for single play combination.'''
    play_1_id: int
    play_2_id: int
    score: float


def match_play(play: Play, possible_matches: List[Play]) -> List[PlayMatch]:
    '''Determine match score for play against all possible_matches.'''
    
    vis_utils.show_plays([play] + possible_matches[0:3])
    
    
    return [PlayMatch(
        play_1_id=play.id,
        play_2_id=other_play.id,
        score=0
    ) for other_play in possible_matches]