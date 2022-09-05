'''Find best match for given play.'''

import sys
from os import path
from typing import List, NamedTuple
sys.path.append(path.join(path.dirname(__file__))) # relative imports are weird

import glog

from play import Play


class PlayMatch(NamedTuple):
    '''Matching score for single play combination.'''
    play_1_id: int
    play_2_id: int
    score: float


def match_play(play: Play, possible_matches: List[Play]) -> List[PlayMatch]:
    '''Determine match score for play against all possible_matches.'''
    
    
    
    
    return [PlayMatch(
        play_1_id=play.id,
        play_2_id=other_play.id,
        score=0
    ) for other_play in possible_matches]