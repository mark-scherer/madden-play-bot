'''Various color utils.'''

from typing import Tuple, Union

import colorsys


def rgb_to_hsv(rgb: Tuple[int, int, int], normalize: bool = True) -> Tuple[int, int, int]:
    '''Converts rgb color to hsv.
    
    Args:
        1. rgb: rgb color as tuple of uint8
        2. normalize: return resulting hsv values 0-1? otherwise 0-255

    Return: hsv color as tuple (normalize = True: 0-1, otherwise uint8)
    '''
    hsv = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
    converted_hsv = hsv if normalize else (
        int(hsv[0]*255),
        int(hsv[1]*255),
        int(hsv[2]*255)
    )
    return converted_hsv


def hsv_to_rgb(
    hsv: Union[Tuple[int, int, int], Tuple[float, float, float]],
    normalized: bool = True
) -> Tuple[int, int, int]:
    '''Converts hsv color to rgb.
    
    Args:
        1. hsv: hsv color as tuple of floats 0-1 (normalized = True) or uint8
        2. normalized: hsv arg normalized as float 0-1? Otherwise uint8

    Return: hsv color as tuple (normalize = True: 0-1, otherwise uint8)
    '''
    assert normalized, 'non-normalized hsv input not yet implemented'

    rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])
    return (
        int(rgb[0]*255),
        int(rgb[1]*255),
        int(rgb[2]*255)
    )


def overlap_semitransparent_color(
    background_color: Tuple[int, int, int],
    foreground_color: Tuple[int, int, int],
    foreground_opacity: float,
    cast_to_int: bool = True
) -> Tuple[int, int, int]:
    '''Return equivalent opaque color after overlaying semitransparent foreground color over opaque background color.'''
    
    def _overlay_channel(background_value: int, foreground_value: int, opacity: float, cast_to_int: bool) -> int:
        overlaid = (opacity*foreground_value) + ((1 - opacity)*background_value)
        return int(overlaid) if cast_to_int else overlaid
    
    assert foreground_opacity >= 0 and foreground_opacity <= 1, \
        'opacity must be 0-1'

    return (
        _overlay_channel(background_color[0], foreground_color[0], foreground_opacity, cast_to_int),
        _overlay_channel(background_color[1], foreground_color[1], foreground_opacity, cast_to_int),
        _overlay_channel(background_color[2], foreground_color[2], foreground_opacity, cast_to_int),
    )