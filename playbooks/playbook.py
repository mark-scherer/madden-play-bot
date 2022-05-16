'''
Playbook dataclasses.
'''

from typing import List, Dict, Any
from enum import Enum
from dataclasses import dataclass

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
    image_url: str
    formation: Formation = None
    type: PlayType = None

    def to_dict(self) -> Dict[str, Any]:
        return self.formation.to_dict() | {
            'play_id': self.id,
            'play_name': self.name,
            'play_image_url': self.image_url,
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