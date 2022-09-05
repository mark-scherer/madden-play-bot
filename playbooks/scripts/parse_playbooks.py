'''
Script for parsing already scraped play images into usable data.

Note: 'scraping' just saves play images.
'''
import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '..')) # upwards relative imports are hacky

import playbooks_parser

PLAYBOOK_DIR = '/tmp/madden-play-bot/playbooks/22-213/'

def main():
    playbooks_parser.parse(playbook_dir=PLAYBOOK_DIR)

main()