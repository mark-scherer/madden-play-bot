'''Play dataclass'''

import sys
from os import path
from typing import List, Dict, Any, NamedTuple, Tuple, Optional
from enum import Enum
from dataclasses import dataclass, field
sys.path.append(path.join(path.dirname(__file__), '..')) # upwards relative imports are hacky

import numpy as np
import cv2
import constants


class FormationFamily(Enum):
    SINGLEBACK = 1
    I_FORM = 2
    STRONG = 3
    WEAK = 4
    GUN = 5
    PISTOL = 7
    HAIL_MARY = 8
    WILDCAT = 9
    GOAL_LINE = 10
    SPECIAL_TEAMS = 11
    KICKOFF = 12
    SAFETY_PUNT = 13


class PlayType(Enum):
    RUN = 1
    PASS = 2


class Point(NamedTuple):
    x: float
    y: float


@dataclass
class PlayMask:
    '''Dataclass for storing play data.
    
    Masks should have play oriented right side up but (0, 0) in pixel coordinates at the top-left, meaning:
    - downfield and the y axis are inverted and offset by ball_location.y

    Assumes x=0 is the left sideline, x=xmax is the right sideline
    '''

    ball_location: Point  # pixel coords of the ball in the image
    mask_local_path: str
    mask: Optional[np.array] = None

    def to_dict(self) -> Dict[str, str]:
        return {
            'playmask_ball_location': f'{self.ball_location.x},{self.ball_location.y}',
            'playmask_local_path': self.mask_local_path
        }

    @staticmethod
    def parse(obj: Dict) -> 'PlayMask':
        ball_location_cords = obj['playmask_ball_location'].split(',')
        mask_local_path = obj['playmask_local_path']
        return PlayMask(
            ball_location=Point(x=ball_location_cords[0], y=ball_location_cords[1]),
            mask_local_path=mask_local_path,
            mask=cv2.imread(mask_local_path, cv2.IMREAD_GRAYSCALE)
        )

    def scale(self) -> float:
        '''Get scale of PlayMask in pixels/yard'''
        _, width = self.mask.shape
        return width / constants.FIELD_WIDTH_YARDS


    @staticmethod
    def save_mask(mask: np.array, filepath: str) -> None:
        cv2.imwrite(filename=filepath, img=mask)


    @staticmethod
    def resample(input_playmask: 'PlayMask', scale: float) -> 'PlayMask':
        '''Resample specified playmask to desired scale and return updated mask.
        
        Args:
            input_playmask: PlayMask to sample, will not be altered
            scale: scale of desired PlayMask in pixels/yard
        '''
        current_mask = input_playmask.mask
        current_height, current_width = current_mask.shape
        current_scale = input_playmask.scale()

        scale_multiplier = scale / current_scale
        new_width = int(scale_multiplier * current_width)
        new_height = int(scale_multiplier * current_height)

        return PlayMask.resize(
            input_playmask=input_playmask,
            new_width=new_width,
            new_height=new_height
        )


    # TODO: might want to skeletonize resized mask - produces a lot of adjacent redundant pixels
        # This just makes later processing less efficient
    @staticmethod
    def resize(input_playmask: 'PlayMask', new_width: int, new_height: int) -> 'PlayMask':
        '''Resizes PlayMask to desired dims, returning copy.'''
        
        current_mask = input_playmask.mask
        current_height, current_width = current_mask.shape
        width_multipler = round(new_width / current_width, 3)
        height_multipler = round(new_height / current_height, 3)
        # assert width_multipler == height_multipler, \
        #     'Width & height not resized equally: ' \
        #     f'width: {current_width} -> {new_width} ({width_multipler}), ' \
        #     f'width: {current_height} -> {new_height} ({height_multipler})'

        interpolation = cv2.INTER_AREA if new_width < current_width else cv2.INTER_LINEAR
        new_mask = cv2.resize(
            src=current_mask, 
            dsize=(new_width, new_height),
            interpolation=interpolation
        )

        new_mask_local_path = input_playmask.mask_local_path
        new_ball_location = Point(
            x=input_playmask.ball_location.x * width_multipler,
            y=input_playmask.ball_location.y * height_multipler,
        )
        PlayMask.save_mask(mask=new_mask, filepath=new_mask_local_path)

        return PlayMask(
            ball_location=new_ball_location,
            mask_local_path=new_mask_local_path,
            mask=new_mask
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
    playmask: Optional[PlayMask] = None

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
        }

    def to_dict(self) -> Dict[str, Any]:
        return  {
            'play_id': self.id,
            'play_name': self.name,
            'play_image_url': self.image_url,
            'play_image_local_path': self.image_local_path,
            'play_formation': self.formation.to_dict(),
            'play_type': str(self.type) if self.type else None,
            'play_playmask': self.playmask.to_dict() if self.playmask else None
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
            playmask = PlayMask.parse(obj['play_playmask']) if obj['play_playmask'] else None
        )