'''
Scrape entire madden playbook.
'''

from typing import List

import glog

import huddle_gg_playbook_scraper as huddle_gg
from playbook import Playbook

MADDEN_YEARS_TO_SCRAPE = [22]
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

def main():
    playbooks = huddle_gg.scrape_playbooks(MADDEN_YEARS_TO_SCRAPE)
    playbooks_to_csv(filepath=PLAYBOOK_CSV_PATH, playbooks=playbooks)

main()