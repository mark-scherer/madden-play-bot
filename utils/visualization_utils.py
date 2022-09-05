'''Various utils for vizualizing results.'''

import sys
from os import path
from typing import List, Dict, Any
import math
sys.path.append(path.join(path.dirname(__file__), '..')) # upwards relative imports are hacky

from matplotlib import pyplot as plt
from matplotlib import patches

from plays.play import Play


def display_images(images: List[Dict[str, Any]]) -> None:
    '''Display debug images in a popup.'''
    plot_cols = None
    plot_rows = None
    if len(images) <= 3:
        plot_rows = 1
    elif len(images) <= 8:
        plot_rows = 2
    else:
        plot_rows = 3
    plot_cols = math.ceil(len(images)/plot_rows)
    for i, display_img_info in enumerate(images):
        plt.subplot(plot_rows, plot_cols, i + 1)
        plt.imshow(display_img_info['img'], cmap='gray')
        plt.title(display_img_info['title'])
        plt.xticks([]), plt.yticks([])

    plt.show()


def show_play(play: Play) -> None:
    '''Display play in a popup.'''

    PLAY_BACKFIELD_YARDS = 10
    PLAY_DOWNFIELD_YARDS = 40
    FIELD_WIDTH_YARDS = 160/3
    YARDLINE_COLOR = 'lightgrey'
    YARDLINE_THICKNESS_PTS = 2
    SIDELINE_THICKNESS_YARDS = 2
    HASMARKS_INSIDE_WIDTH_YARDS = 18.6/3
    HASHMARKS_LENGTH_YARDS = 2/3
    YARD_MARKS_CENTER_DISTANCE_TO_SIDELINE_YARDS = (12 + 9) / 2

    PLAY_COLOR = 'black'
    BALL_SIZE_PTS = 4

    field_xmin = -1 * FIELD_WIDTH_YARDS / 2
    field_xmax = FIELD_WIDTH_YARDS / 2
    field_ymin = -1*PLAY_BACKFIELD_YARDS
    field_ymax = PLAY_DOWNFIELD_YARDS
    
    # set title and axis limits and clear markers
    fig, ax = plt.subplots()
    ax.set_xlim(field_xmin, field_xmax)
    ax.set_ylim(field_ymin, field_ymax)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(play.title())

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
    yard_markers_x = FIELD_WIDTH_YARDS/2 - YARD_MARKS_CENTER_DISTANCE_TO_SIDELINE_YARDS
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