'''
Playbook dataclasses.
'''

import sys
from os import path
from typing import List, Dict, Any
from enum import Enum
from dataclasses import dataclass
import json
sys.path.append(path.join(path.dirname(__file__), '..', 'plays')) # upwards relative imports are hacky

import glog

from play import Play


class PlaybookType(Enum):
    OFFENSE = 1
    DEFENSE = 2
    ALTERNATE = 3


@dataclass
class Playbook:
    id: int
    name: str
    type: PlaybookType
    madden_year: int = None
    plays: List[Play] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'playbook_id': self.id,
            'playbook_name': self.name,
            'playbook_type': str(self.type),
            'playbook_madden_year': self.madden_year,
            'playbook_plays': [play.to_dict() for play in self.plays]
        }

    @staticmethod
    def parse(obj: Dict) -> 'Playbook':
        type_str = obj['playbook_type'].split('.')[1]
        return Playbook(
            id = obj['playbook_id'],
            name = obj['playbook_name'],
            type = PlaybookType[type_str],
            madden_year = obj['playbook_madden_year'],
            plays = [Play.parse(play) for play in obj['playbook_plays']]
        )
    

    @staticmethod
    def write_playbooks_to_json(filepath: str, playbooks: List['Playbook']) -> None:
        '''Write list of playbooks to json.'''
        playbooks_dict = [pb.to_dict() for pb in playbooks]
        with open(filepath, 'w') as f:
            f.write(json.dumps(playbooks_dict))
        glog.info(f'wrote {len(playbooks)} playbooks to {filepath}')

    @staticmethod
    def read_playbooks_from_json(filepath: str) -> List['Playbook']:
        playbooks = []
        with open(filepath, 'r') as f:
            playbooks_dict = json.loads(f.read())
            playbooks = [Playbook.parse(playbook_obj) for playbook_obj in playbooks_dict]
        return playbooks
            
