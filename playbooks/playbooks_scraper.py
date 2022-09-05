'''
Functions for scraping & saving entire madden playbook from generic source.
'''

from typing import List
import pathlib
from os import path

import glog
import requests

import playbooks_scraper_huddle_gg as huddle_gg
from playbook import Playbook
import constants


def _download_image(url: str, local_path: str, quiet: bool = False) -> None:
        with open(local_path, 'wb') as f:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            for block in response.iter_content(1024):
                if not block:
                    break
                f.write(block)
        if not quiet:
            glog.info(f'downloaded {url} to {local_path}.')


def _download_play_images(playbook_dir: str, playbook: Playbook) -> Playbook:
    '''Download play images to local for all plays in specified playbook, 
    return playbook with image_local_path populated.
    '''

    glog.info(f'attempting to download playbook {playbook.title()} ({len(playbook.plays)} plays) to {playbook_dir}..')

    downloaded_play_images = 0
    pathlib.Path(playbook_dir).mkdir(parents=True, exist_ok=True)
    play_images_dir = path.join(playbook_dir, constants.PLAY_IMAGES_SUBDIR)
    pathlib.Path(play_images_dir).mkdir(parents=True, exist_ok=True)

    for play in playbook.plays:
        image_ext = path.splitext(play.image_url)[1]
        play_filename = f'{play.formation.id}-{play.id}'
        
        if image_ext:
            play_filename += image_ext # image_ext includes '.': .<ext>

        play_filepath = path.join(play_images_dir, play_filename)
        _download_image(url=play.image_url, local_path=play_filepath)
        
        play.image_local_path = play_filepath
        downloaded_play_images += 1

    glog.info(f'..downloaded {downloaded_play_images} play images to {play_images_dir}')
    return playbook


def scrape(
    madden_year: int,
    playbook_id: int,
    formation_id: int
) -> Playbook:
    '''Downloads plays to local json and saves images.
    
    Args:
        madden_year: madden playbook year to scrape
        playbook_id: huddle_gg playbook_id to scrape
        formation_id: huddle_gg formation_id to scrape
    '''

    playbook = huddle_gg.scrape_playbook(
        madden_year=madden_year,
        playbook_id=playbook_id,
        formation_id=formation_id
    )

    playbook_dir = path.join(constants.PLAYBOOKS_BASE_DIR, f'{playbook.madden_year}-{playbook.id}')

    playbook = _download_play_images(playbook_dir=playbook_dir, playbook=playbook)
    scraped_playbook_filepath = path.join(playbook_dir, constants.SCRAPED_PLAYBOOK_DATA_FILENAME)
    playbook.write_to_json(filepath=scraped_playbook_filepath)