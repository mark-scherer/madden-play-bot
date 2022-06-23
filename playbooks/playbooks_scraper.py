'''
Functions for scraping & saving entire madden playbook from generic source.
'''

from typing import List
import pathlib
import os

import glog
import requests

import playbooks_scraper_huddle_gg as huddle_gg
from playbook import Playbook
import constants

MADDEN_YEARS_TO_SCRAPE = [22]

def _download_play_images(output_dir: str, playbooks: List[Playbook]) -> List[Playbook]:
    '''Download play images to local for all plays in specified playbook, 
    return playbook with image_local_path populated.
    '''

    def download_image(url: str, local_path: str, quiet: bool = False) -> None:
        with open(local_path, 'wb') as f:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            for block in response.iter_content(1024):
                if not block:
                    break
                f.write(block)
        if not quiet:
            glog.info(f'downloaded {url} to {local_path}.')

    glog.info(f'attempting to download play images for {len(playbooks)} playbooks to {output_dir}...')

    downloaded_play_images = 0
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    for playbook in playbooks:
        playbook_dir = os.path.join(output_dir, f'{playbook.madden_year}-{playbook.id}')
        pathlib.Path(playbook_dir).mkdir(parents=True, exist_ok=True)
        for play in playbook.plays:
            image_ext = os.path.splitext(play.image_url)[1]
            play_filename = f'{play.formation.id}-{play.id}'
            if image_ext:
                play_filename += image_ext # image_ext includes '.': .<ext>
            play_filepath = os.path.join(playbook_dir, play_filename)
            download_image(url=play.image_url, local_path=play_filepath)
            play.image_local_path = play_filepath
            downloaded_play_images += 1
    glog.info(f'downloaded {downloaded_play_images} play images from {len(playbooks)} playbooks to {output_dir}')
    return playbooks

def scrape():
    '''For now just write plays to csv & download play images locally.'''

    playbooks = huddle_gg.scrape_playbooks(MADDEN_YEARS_TO_SCRAPE)
    playbooks = _download_play_images(output_dir=constants.PLAYBOOK_IMAGES_DIR, playbooks=playbooks)
    Playbook.write_playbooks_to_json(filepath=constants.SCRAPED_PLAYBOOK_PATH, playbooks=playbooks)