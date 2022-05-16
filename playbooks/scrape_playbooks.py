'''
Scrape entire madden playbook.
'''

from typing import List
import pathlib
import os

import glog
import requests

import playbooks_scraper_huddle_gg as huddle_gg
from playbook import Playbook

MADDEN_YEARS_TO_SCRAPE = [22]
PLAYBOOK_IMAGES_DIR = '/Users/mark/Downloads/scraped_madden_play_images/'
PLAYBOOK_CSV_PATH = '/Users/mark/Downloads/scraped_madden_plays.csv'
PLAYBOOK_CSV_HEADERS = [
    'playbook_madden_year',
    'playbook_type',
    'playbook_id',
    'playbook_name',
    'formation_family',
    'formation_id',
    'formation_name',
    'play_type',
    'play_id',
    'play_name',
    'play_image_url',
]

def playbooks_to_csv(filepath: str, playbooks: List[Playbook]) -> None:
    written_play_count = 0
    with open(filepath, 'w') as f:
        f.write(','.join(PLAYBOOK_CSV_HEADERS) + '\n')
        for playbook in playbooks:
            for play in playbook.to_play_dicts():
                f.write(','.join([
                    str(play[col]) for col in PLAYBOOK_CSV_HEADERS
                ]) + '\n')
                written_play_count += 1
    glog.info(f'wrote {written_play_count} plays from {len(playbooks)} playbooks to {filepath}.')

def download_play_images(output_dir: str, playbooks: List[Playbook]) -> None:
    # for now just download to local dir

    def download_image(url: str, local_path: str) -> None:
        with open(local_path, 'wb') as f:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            for block in response.iter_content(1024):
                if not block:
                    break
                f.write(block)
        glog.info(f'downloaded {url} to {local_path}.')

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
            downloaded_play_images += 1
    glog.info(f'downloaded {downloaded_play_images} play images from {len(playbooks)} playbooks to {output_dir}')


def main():
    playbooks = huddle_gg.scrape_playbooks(MADDEN_YEARS_TO_SCRAPE)
    playbooks_to_csv(filepath=PLAYBOOK_CSV_PATH, playbooks=playbooks)
    download_play_images(output_dir=PLAYBOOK_IMAGES_DIR, playbooks=playbooks)

main()