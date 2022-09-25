'''
Script for scraping & saving madden playbooks.

Note: 'scraping' just saves play images, it does not 'parse' them into usable data.
'''

import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '..')) # upwards relative imports are hacky

import playbooks_scraper


MADDEN_YEAR = 22
PLAYBOOK_ID = 214
FORMATION_ID = 4669 


def main():
    playbooks_scraper.scrape(
        madden_year=MADDEN_YEAR,
        playbook_id=PLAYBOOK_ID,
        formation_id=FORMATION_ID
    )

main()