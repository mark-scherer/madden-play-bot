'''Various utils for vizualizing results.'''

import sys
from os import path
from typing import List, Dict, Any, Tuple
import math
sys.path.append(path.join(path.dirname(__file__), '..')) # upwards relative imports are hacky

from matplotlib import pyplot as plt
from matplotlib import patches

import constants
from plays.play import Play


def _get_subplot_dims(figure_count: int) -> Tuple[int, int]:
    '''Determine number of rows and columns in subplot'''
    rows = None
    cols = None

    if figure_count <= 3:
        rows = 1
    elif figure_count <= 8:
        rows = 2
    else:
        rows = 3
    cols = math.ceil(figure_count/rows)

    return (rows, cols)



def display_images(images: List[Dict[str, Any]]) -> None:
    '''Display debug images in a popup.'''
    rows, cols = _get_subplot_dims(len(images))
    for i, display_img_info in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(display_img_info['img'], cmap='gray')
        plt.title(display_img_info['title'])
        plt.xticks([]), plt.yticks([])

    plt.show()


def show_plays(plays: List[Play]) -> None:
    '''Display plays in a popup.'''

    PLAY_BACKFIELD_YARDS = 10
    PLAY_DOWNFIELD_YARDS = 40
    YARDLINE_COLOR = 'lightgrey'
    YARDLINE_THICKNESS_PTS = 2
    SIDELINE_THICKNESS_YARDS = 2
    HASMARKS_INSIDE_WIDTH_YARDS = 18.6/3
    HASHMARKS_LENGTH_YARDS = 2/3
    YARD_MARKS_CENTER_DISTANCE_TO_SIDELINE_YARDS = (12 + 9) / 2

    PLAY_COLOR = 'black'
    BALL_SIZE_PTS = 4

    field_xmin = -1 * constants.FIELD_WIDTH_YARDS / 2
    field_xmax = constants.FIELD_WIDTH_YARDS / 2
    field_ymin = -1*PLAY_BACKFIELD_YARDS
    field_ymax = PLAY_DOWNFIELD_YARDS

    rows, cols = _get_subplot_dims(len(plays))
    fig, _ = plt.subplots(rows, cols)
    
    # set title and axis limits and clear markers
    for i, play in enumerate(plays):
        ax = plt.subplot(rows, cols, i+1)
        ax.set_xlim(field_xmin, field_xmax)
        ax.set_ylim(field_ymin, field_ymax)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        total_points = sum([len(route.points) for route in play.routes])
        ax.set_title(f'{play.title()} (points: {total_points})')

        # Draw yard lines
        yard_lines_y = range(field_ymin, field_ymax + 1, 5)
        ax.hlines(y=yard_lines_y, xmin=field_xmin, xmax=field_xmax, color=YARDLINE_COLOR, linewidth=YARDLINE_THICKNESS_PTS)

        # Draw sidelines
        sideline_height = field_ymax - field_ymin
        left_sideline = patches.Rectangle(
            xy=(field_xmin - SIDELINE_THICKNESS_YARDS, field_ymin),
            width=SIDELINE_THICKNESS_YARDS,
            height=sideline_height,
            color=YARDLINE_COLOR    
        )
        right_sideline = patches.Rectangle(
            xy=(field_xmax, field_ymin),
            width=SIDELINE_THICKNESS_YARDS,
            height=sideline_height,
            color=YARDLINE_COLOR    
        )
        ax.add_patch(left_sideline)
        ax.add_patch(right_sideline)

        # Draw hashes
        hashmarks_xmin = HASMARKS_INSIDE_WIDTH_YARDS/2
        hashmarks_xmax = hashmarks_xmin + HASHMARKS_LENGTH_YARDS
        hashmarks_y = range(field_ymin, field_ymax + 1, 1)
        for y in hashmarks_y:
            ax.hlines(y=y, xmin=hashmarks_xmin, xmax=hashmarks_xmax, color=YARDLINE_COLOR, linewidth=YARDLINE_THICKNESS_PTS)
            ax.hlines(y=y, xmin=-1*hashmarks_xmin, xmax=-1*hashmarks_xmax, color=YARDLINE_COLOR, linewidth=YARDLINE_THICKNESS_PTS)

        # Draw numbers
        yard_markers_y = range(field_ymin, field_ymax + 1, 10)
        yard_markers_x = constants.FIELD_WIDTH_YARDS/2 - YARD_MARKS_CENTER_DISTANCE_TO_SIDELINE_YARDS
        for y in yard_markers_y:
            ax.text(x=yard_markers_x, y=y, s=y, 
                ha='center', va='center', color=YARDLINE_COLOR,
                rotation='vertical', size='x-large')
            ax.text(x=-1*yard_markers_x, y=y, s=y, 
                ha='center', va='center', color=YARDLINE_COLOR,
                rotation='vertical', size='x-large')

        # Draw ball
        ax.plot(0, 0, 's', color=PLAY_COLOR, ms=BALL_SIZE_PTS)

        # Draw routes
        for route in play.routes:
            pt_x = [pt.x for pt in route.points]
            pt_y = [pt.y for pt in route.points]
            ax.plot(pt_x, pt_y, color=PLAY_COLOR, marker='.', linestyle='none')

    plt.show()