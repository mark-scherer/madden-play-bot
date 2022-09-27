'''Play dataclass'''

import sys
from os import path
from typing import List, Dict, Any, NamedTuple, Tuple, Optional
from enum import Enum
from dataclasses import dataclass, field
sys.path.append(path.join(path.dirname(__file__), '..')) # upwards relative imports are hacky

import numpy as np
import cv2
import skimage
from skimage.morphology import skeletonize

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


class SpacingCorrection(NamedTuple):
    '''Object for storing a formation's spacing correction due to inconsistent horizontal madden play spacing.
    Madden plays upscale space within the core of the formation, causing reduced spacing outside the TEs.
    '''
    original_xmin: float
    original_xmax: float
    corrected_xmin: float
    corrected_xmax: float


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
            ball_location=Point(x=float(ball_location_cords[0]), y=float(ball_location_cords[1])),
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

        return PlayMask._resize(
            input_playmask=input_playmask,
            new_width=new_width,
            new_height=new_height
        )


    # TODO: might want to skeletonize resized mask - produces a lot of adjacent redundant pixels
        # This just makes later processing less efficient
    @staticmethod
    def _resize(input_playmask: 'PlayMask', new_width: int, new_height: int) -> 'PlayMask':
        '''Resizes PlayMask to desired dims, returning copy.
        
        Note: external methods should probably use resample().
        '''
        
        current_mask = input_playmask.mask
        current_height, current_width = current_mask.shape
        width_multipler = round(new_width / current_width, 3)
        height_multipler = round(new_height / current_height, 3)
        # assert width_multipler == height_multipler, \
        #     'Width & height not resized equally: ' \
        #     f'width: {current_width} -> {new_width} ({width_multipler}), ' \
        #     f'width: {current_height} -> {new_height} ({height_multipler})'

        interpolation = constants.DOWNSIZE_INTERPOLATION if new_width < current_width else constants.UPSIZE_INTERPOLATION
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


    @staticmethod
    def crop_field_vertically(
        input_playmask: 'PlayMask', 
        backfield_yards: int = constants.PLAYMASK_BACKFIELD_YARDS,
        downfield_yards: int = constants.PLAYMASK_DOWNFIELD_YARDS
    ) -> 'PlayMask':
        '''Crop PlayMask in y direction while maintaining current scale.
        Modifies & returns copy.
        '''

        current_mask = input_playmask.mask
        current_ball_y = input_playmask.ball_location.y
        current_height, width = current_mask.shape
        scale = input_playmask.scale()
        
        current_backfield_yards = (current_height - current_ball_y) / scale
        current_downfield_yards = current_ball_y / scale

        new_height = int((backfield_yards + downfield_yards) * scale)
        new_ball_y = int(new_height * downfield_yards / (downfield_yards + backfield_yards))
        new_ball_location = Point(
            x=input_playmask.ball_location.x,
            y=new_ball_y
        )
        new_mask = np.zeros((new_height, width), np.uint8)

        # Copy over PlayMask pixels
        copied_pixels_current_ymin = None
        copied_pixels_new_ymin = None
        copied_pixels_current_ymax = None
        copied_pixels_new_ymax = None

        # Less downfield shown, must crop playmask to new downfield limit
        if downfield_yards < current_downfield_yards:
            copied_pixels_current_ymin = round(current_ball_y - (downfield_yards * scale))
            copied_pixels_new_ymin = 0
        
        # More downfield shown, must add black pixels to top of playmask
        else:
            copied_pixels_current_ymin = 0
            copied_pixels_new_ymin = round(new_ball_y - (current_downfield_yards * scale))

        copied_pixels_height = min(
            current_height - copied_pixels_current_ymin,
            new_height - copied_pixels_new_ymin
        )
        copied_pixels_current_ymax = copied_pixels_current_ymin + copied_pixels_height
        copied_pixels_new_ymax = copied_pixels_new_ymin + copied_pixels_height

        new_mask[copied_pixels_new_ymin:copied_pixels_new_ymax, 0:width-1] = \
            current_mask[copied_pixels_current_ymin:copied_pixels_current_ymax, 0:width-1]

        # Save and return mask
        new_mask_local_path = input_playmask.mask_local_path
        PlayMask.save_mask(mask=new_mask, filepath=new_mask_local_path)
        return PlayMask(
            ball_location=new_ball_location,
            mask_local_path=new_mask_local_path,
            mask=new_mask
        )


    @staticmethod
    def _stretch_horizontal_midpoint(
        current_mask: np.array,
        current_midpoint_x: int, 
        new_midpoint_x: int
    ) -> np.array:
        '''Adjusts content of mask horizontally by moving midpoint as specified.
        
        Note:
        - Splits mask at midpoint then rescales left & right halves to achieve new midpoint.
        - Returned mask will have the same dimensions.
        '''

        height, width = current_mask.shape
        assert current_midpoint_x > 0 and current_midpoint_x < width and \
            new_midpoint_x > 0 and new_midpoint_x < width, \
            f'attempting to move midpoint outside mask xbounds [0-{width}): {current_midpoint_x} -> {new_midpoint_x}'

        current_left_crop = current_mask[0:height-1, 0:current_midpoint_x]
        current_right_crop = current_mask[0:height-1, current_midpoint_x+1:width-1]

        new_left_crop_width = new_midpoint_x
        new_right_crop_width = width - new_midpoint_x

        print(f'rescaling midpoint from {current_midpoint_x} to {new_midpoint_x}')

        new_left_crop = cv2.resize(
            src=current_left_crop, 
            dsize=(new_left_crop_width, height),
            interpolation=constants.DOWNSIZE_INTERPOLATION
        )
        new_right_crop = cv2.resize(
            src=current_right_crop, 
            dsize=(new_right_crop_width, height),
            interpolation=constants.DOWNSIZE_INTERPOLATION
        )

        new_mask = np.zeros((height, width), np.uint8)
        new_mask[0:height, 0:new_midpoint_x] = new_left_crop
        new_mask[0:height, new_midpoint_x:width] = new_right_crop

        return new_mask


    @staticmethod
    def recenter_ball_horizontally(input_playmask: 'PlayMask', ball_x_frac: int) -> 'PlayMask':
        '''Adjust PlayMask horizontally so ball moved to desired x location, modifying and returning copy.

        Applies a single spacing adjustment by moving the center of the play laterally.
        
        Note: 
        - Splits play vertically at ball and scales left and right havles independently.
        - Returned PlayMask will have the same dimensions and avg scale, although actual horiztonal scale will now be disjoint.
        '''

        inside_hashes_width_frac = constants.HASMARKS_INSIDE_WIDTH_YARDS / constants.FIELD_WIDTH_YARDS
        inside_hashes_xmin = 0.5 - inside_hashes_width_frac/2
        inside_hashes_xmax = 0.5 + inside_hashes_width_frac/2

        assert ball_x_frac >= inside_hashes_xmin and ball_x_frac <= inside_hashes_xmax, \
            f'ball_x_frac {ball_x_frac} is outside hashes ({round(inside_hashes_xmin, 3)} - {round(inside_hashes_xmax, 3)})'

        avg_scale = input_playmask.scale()
        current_mask = input_playmask.mask
        height, width = current_mask.shape
        current_ball_x = round(input_playmask.ball_location.x)

        new_ball_x = round(width*ball_x_frac)
        new_mask = PlayMask._stretch_horizontal_midpoint(
            current_mask=current_mask,
            current_midpoint_x=current_ball_x,
            new_midpoint_x=new_ball_x
        )

        new_ball_location = Point(
            x=new_ball_x,
            y=input_playmask.ball_location.y
        )

        print(f'recentered ball from {input_playmask.ball_location} to {new_ball_location}')

        # Save and return mask
        new_mask_local_path = input_playmask.mask_local_path
        PlayMask.save_mask(mask=new_mask, filepath=new_mask_local_path)
        return PlayMask(
            ball_location=new_ball_location,
            mask_local_path=new_mask_local_path,
            mask=new_mask
        )


    @staticmethod
    def apply_horizontal_spacing_correction(input_playmask: 'PlayMask', correction: SpacingCorrection) -> 'PlayMask':
        '''Adjust PlayMask horizontally so width of formation core adjusted.
        
        Increases or decreases width of formation core - necessary b/c madden playmasks
        do not use consistent spacing in the core of the formation.
        
        Notes:
        - Scales left and right sides of formation horizontally - ball will remain in the same position.
        - Splits play into forths horizontally and rescales two middle forths, stretching outer thirds as necessary.
        - Returned PlayMask will have the same dimensions and avg scale.
        '''
        current_mask = input_playmask.mask
        ball_x = int(input_playmask.ball_location.x)
        height, width = current_mask.shape

        assert correction.original_xmin < ball_x and correction.corrected_xmin < ball_x, \
            f'correction xmins must be left of ball (x: {ball_x}), attempting to scale {correction.original_xmin} -> {correction.corrected_xmin}'
        assert correction.original_xmax > ball_x and correction.corrected_xmax > ball_x, \
            f'correction xmaxes must be right of ball (x: {ball_x}), attempting to scale {correction.original_xmax} -> {correction.corrected_xmax}'

        current_left_mask = current_mask[:, 0:ball_x]
        new_left_mask = PlayMask._stretch_horizontal_midpoint(
            current_mask=current_left_mask,
            current_midpoint_x=round(correction.original_xmin),
            new_midpoint_x=round(correction.corrected_xmin)
        )

        current_right_mask = current_mask[:, ball_x:width]
        new_right_mask = PlayMask._stretch_horizontal_midpoint(
            current_mask=current_right_mask,
            current_midpoint_x=round(correction.original_xmax - ball_x),
            new_midpoint_x=round(correction.corrected_xmax - ball_x)
        )

        new_mask = np.zeros((height, width), np.uint8)
        new_mask[0:height, 0:ball_x] = new_left_mask
        new_mask[0:height, ball_x:width] = new_right_mask

        # Save and return mask
        new_mask_local_path = input_playmask.mask_local_path
        PlayMask.save_mask(mask=new_mask, filepath=new_mask_local_path)
        return PlayMask(
            ball_location=input_playmask.ball_location,
            mask_local_path=new_mask_local_path,
            mask=new_mask
        )

    
    def apply_backfield_vertical_scaling(input_playmask: 'PlayMask', backfield_scaling_factor: float) -> 'PlayMask':
        '''Apply vertical scaling correction to just the backfield.'''
        
        current_mask = input_playmask.mask
        current_height, width = current_mask.shape
        ball_y = int(input_playmask.ball_location.y)
        downfield_crop = current_mask[0:ball_y, 0:width]
        current_backfield_crop = current_mask[ball_y+1:current_height-1, 0:width-1]

        current_backfield_height = current_height - ball_y
        new_backfield_height = int(current_backfield_height * backfield_scaling_factor)

        interpolation = constants.DOWNSIZE_INTERPOLATION if new_backfield_height < current_backfield_height else constants.UPSIZE_INTERPOLATION
        new_backfield_crop = cv2.resize(
            src=current_backfield_crop, 
            dsize=(width, new_backfield_height),
            interpolation=interpolation
        )

        new_height = ball_y + new_backfield_height
        new_mask = np.zeros((new_height, width), np.uint8)
        new_mask[0:ball_y, 0:width] = downfield_crop
        new_mask[ball_y:new_height, 0:width] = new_backfield_crop

        # Save and return mask
        new_mask_local_path = input_playmask.mask_local_path
        PlayMask.save_mask(mask=new_mask, filepath=new_mask_local_path)
        return PlayMask(
            ball_location=input_playmask.ball_location,
            mask_local_path=new_mask_local_path,
            mask=new_mask
        )


    # This does not work for some reason
    @staticmethod
    def reskeletonize(input_playmask: 'PlayMask') -> 'PlayMask':
        '''Reskeletonize PlayMask - modifies & returns copy.'''

        raise NotImplementedError('this method does not work, need to fix')

        current_mask = input_playmask.mask
        skeltonized = skeletonize(skimage.img_as_bool(current_mask.copy()))
        new_mask = skeltonized.astype('uint8') * 255

        print(f'reskeletonized {current_mask.dtype}, {current_mask.shape}, {np.sum(np.concatenate(current_mask))} pixels to {new_mask.dtype}, {new_mask.shape} {np.sum(np.concatenate(new_mask))} pixels')

        new_mask_local_path = input_playmask.mask_local_path
        new_ball_location = input_playmask.ball_location
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
    mask: Optional[PlayMask] = None
    spacing_correction: Optional[SpacingCorrection] = None

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
            'formation_mask': self.mask.to_dict() if self.mask else None,
            'formation_correction': self.spacing_correction
        }

    @staticmethod
    def parse(obj: Dict) -> 'Formation':
        family_str = obj['formation_family'].split('.')[1]
        return Formation(
            id = obj['formation_id'],
            name = obj['formation_name'],
            family = FormationFamily[family_str],
            mask = PlayMask.parse(obj['formation_mask']) if obj['formation_mask'] else None,
            spacing_correction = obj['formation_correction']
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