'''Play dataclass'''

from typing import List, Dict, Any, NamedTuple
from enum import Enum
from dataclasses import dataclass, field


class FormationFamily(Enum):
    SINGLEBACK = 1
    I_FORM = 2
    STRONG = 3
    WEAK = 4
    GUN = 5
    PISTOL = 7
    HAIL_MARY = 8
    GOAL_LINE = 9
    SPECIAL_TEAMS = 10
    KICKOFF = 11
    SAFETY_PUNT = 12


class PlayType(Enum):
    RUN = 1
    PASS = 2


class Point(NamedTuple):
    x: float
    y: float


@dataclass
class Route:
    points: List[Point] = field(default_factory=list) # must do this to default = []

    def summary(self) -> Dict[str, Any]:
        '''Generate summary dict.'''
        return {
            'point_count': len(self.points)
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            'route_points': [(point.x, point.y) for point in self.points]
        }

    @staticmethod
    def parse(obj: Dict) -> 'Route':
        return Route(
            points = [Point(x=pt[0], y=pt[1]) for pt in obj['route_points']]
        )


@dataclass
class Formation:
    id: str
    name: str
    family: FormationFamily

    def title(self) -> str:
        '''Generate short title string.'''
        return self.name
    
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
            'formation_family': str(self.family),
        }

    @staticmethod
    def parse(obj: Dict) -> 'Formation':
        family_str = obj['formation_family'].split('.')[1]
        return Formation(
            id = obj['formation_id'],
            name = obj['formation_name'],
            family = FormationFamily[family_str]
        )


@dataclass
class Play:
    id: int
    name: str
    image_url: str = None
    image_local_path: str = None
    formation: Formation = None
    type: PlayType = None
    routes: List[Route] = field(default_factory=list) # must do this to default = []

    def title(self) -> str:
        '''Generate short title string.'''
        return f'{self.formation.title()}: {self.name}'

    
    def summary(self) -> Dict[str, Any]:
        '''Generate summary dict.'''
        return {
            'formation': self.formation.summary(),
            'name': self.name,
            'id': self.id,
            'type': self.type,
            'route_count': len(self.routes),
            'routes': [route.summary() for route in self.routes]
        }

    def to_dict(self) -> Dict[str, Any]:
        return  {
            'play_id': self.id,
            'play_name': self.name,
            'play_image_url': self.image_url,
            'play_image_local_path': self.image_local_path,
            'play_formation': self.formation.to_dict(),
            'play_type': str(self.type) if self.type else None,
            'play_routes': [route.to_dict() for route in self.routes]
        }

    @staticmethod
    def parse(obj: Dict) -> 'Play':
        type_str = obj['play_type'].split('.')[1] if obj['play_type'] else None
        return Play(
            id = obj['play_id'],
            name = obj['play_name'],
            image_url = obj['play_image_url'],
            image_local_path = obj['play_image_local_path'],
            formation = Formation.parse(obj['play_formation']),
            type = PlayType[type_str] if type_str else None,
            routes = [Route.parse(route) for route in obj['play_routes']]
        )