'''
Playbook dataclasses.
'''

from typing import List, Dict, Any
from enum import Enum
from dataclasses import dataclass
import csv

import glog

from constants import PLAYBOOK_CSV_HEADERS

class FormationFamily(Enum):
    SINGLEBACK = 1
    I_FORM = 2
    STRONG = 3
    WEAK = 4
    GUN = 5
    GOAL_LINE = 6
    HAIL_MARY = 7
    SPECIAL_TEAMS = 8
    KICKOFF = 9
    SAFETY_PUNT = 10

class PlayType(Enum):
    RUN = 1
    PASS = 2

class PlaybookType(Enum):
    '''Playbook types'''
    OFFENSE = 1
    DEFENSE = 2
    ALTERNATE = 3

@dataclass
class Formation:
    id: str
    name: str
    family: FormationFamily

    def summary(self) -> Dict[str, str]:
        '''Generate summary dict.'''
        return {
            'family': str(self.family),
            'name': self.name,
            'id': self.id,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            'formation_id': self.id,
            'formation_name': self.name,
            'formation_family': self.family
        }

@dataclass
class Play:
    id: int
    name: str
    image_url: str = None
    image_local_path: str = None
    formation: Formation = None
    type: PlayType = None

    def summary(self) -> Dict[str, Any]:
        '''Generate summary dict.'''
        return {
            'formation': self.formation.summary(),
            'name': self.name,
            'id': self.id,
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.formation.to_dict() | {
            'play_id': self.id,
            'play_name': self.name,
            'play_image_url': self.image_url,
            'play_image_local_path': self.image_local_path,
            'play_type': self.type,
        }

@dataclass
class Playbook:
    id: int
    name: str
    type: PlaybookType
    madden_year: int = None
    plays: List[Play] = None

    def to_play_dicts(self) -> List[Dict[str, Any]]:
        return [
            {
                'playbook_id': self.id,
                'playbook_name': self.name,
                'playbook_type': self.type,
                'playbook_madden_year': self.madden_year,
            } | play.to_dict()
            for play in self.plays
        ]

    @staticmethod
    def playbooks_to_csv(filepath: str, playbooks: List['Playbook']) -> None:
        '''Write list of playbooks to csv.'''
        
        written_play_count = 0
        with open(filepath, 'w') as f:
            f.write(','.join(PLAYBOOK_CSV_HEADERS) + '\n')
            for playbook in playbooks:
                for play in playbook.to_play_dicts():
                    f.write(','.join([
                        str(play[col]) for col in PLAYBOOK_CSV_HEADERS
                    ]) + '\n')
                    written_play_count += 1
        glog.info(f'wrote {written_play_count} plays from {len(playbooks)} playbooks to {filepath}.')
    
    @staticmethod
    def playbooks_from_csv(filepath: str) -> List['Playbook']:
        '''Read list of playbooks from csv.'''

        play_dicts = []
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row == PLAYBOOK_CSV_HEADERS:
                    continue
                play_dict = dict(zip(PLAYBOOK_CSV_HEADERS, row))
                play_dicts.append(play_dict)

        playbooks_dict = {}
        read_plays_count = 0
        for play_dict in play_dicts:
            playbook_id = play_dict['playbook_id']
            if playbook_id not in playbooks_dict:
                playbook_type_str = play_dict['playbook_type'].split('.')[1]
                playbooks_dict[playbook_id] = Playbook(
                    id=playbook_id,
                    name=play_dict['playbook_name'],
                    type=PlaybookType[playbook_type_str],
                    madden_year=play_dict['playbook_madden_year'],
                    plays=[]
                )
            playbook = playbooks_dict[playbook_id]
            
            formation_family_str = play_dict['formation_family'].split('.')[1]
            formation = Formation(
                id=play_dict['formation_id'],
                name=play_dict['formation_name'],
                family=FormationFamily[formation_family_str],
            )
            
            play_type_str = play_dict['play_type'].split('.')[1] if play_dict['play_type'] != 'None' else None
            playbook_type = PlayType[play_type_str] if play_type_str else None
            play = Play(
                id=play_dict['play_id'],
                name=play_dict['play_name'],
                type=playbook_type,
                image_url=play_dict['play_image_url'],
                image_local_path=play_dict['play_image_local_path'],
                formation=formation,
            )
            playbook.plays.append(play)
            read_plays_count += 1
        
        playbooks = playbooks_dict.values()
        glog.info(f'read {read_plays_count} plays from {len(playbooks)} playbooks from {filepath}')
        return playbooks