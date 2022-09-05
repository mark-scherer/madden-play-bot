'''
Functions for scraping entire madden playbook from hustle.gg.
'''

import sys
from os import path
from typing import List, Dict, NamedTuple
import os
import json
sys.path.append(path.join(path.dirname(__file__), '..')) # upwards relative imports are hacky

import requests
import glog
from bs4 import BeautifulSoup

from plays.play import Play, Formation, FormationFamily
from playbook import Playbook, PlaybookType

SUPPORTED_YEARS = [21, 22]

class PlayScrapingInfo(NamedTuple):
    name: str
    image_url: str


def _playbooks_dir_url(madden_year: int) -> str:
    '''Return url for base dir for madden year, which will show all available playbooks'''

    assert madden_year in SUPPORTED_YEARS, f'madden_year {madden_year} not supported - only supports {SUPPORTED_YEARS}'
    return f'https://huddle.gg/{madden_year}/playbooks/'


def _playbook_url(playbook: Playbook) -> str:
    '''Return base url for given playbook'''

    return os.path.join(_playbooks_dir_url(playbook.madden_year), str(playbook.id))


def _formation_url(playbook: Playbook, formation: Formation) -> str:
    return os.path.join(_playbook_url(playbook), 'formations', str(formation.id))


def _scrape_formation_plays(playbook: Playbook, formation: Formation) -> List[Play]:
    '''Return list of scraped plays given playbook and formation.'''
    formation_url = _formation_url(playbook, formation)
    response = requests.get(formation_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    plays = []
    for play_element in soup.find_all(class_='play-tile'):
        name = play_element.text.replace('\n', '')
        style = play_element.find(class_='play-tile__image')['style']
        image_url = style.split('(')[1].split(')')[0]
        link = play_element['href']
        link = link[:-1] if link.endswith('/') else link
        id_ = int(os.path.split(link)[1])

        plays.append(Play(
            id=id_, 
            name=name,
            image_url=image_url,
            formation=formation
        ))

    return plays


def _scrape_available_formations(playbook: Playbook) -> Dict[int, Formation]:
    '''Return list of scraped formations given playbook.'''
    playbook_url = _playbook_url(playbook)
    response = requests.get(playbook_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    
    available_formations = {}
    for formations_list in soup.find_all(class_='playbooks-list'):
        family_str = formations_list.find_previous_sibling('h3').text
        family = FormationFamily[family_str.upper().replace(' ', '_')]

        for formation_element in formations_list.find_all(class_='playbooks-list__item'):
            name = formation_element.text.replace('\n', '')
            link = formation_element.find('a')['href']
            link = link[:-1] if link.endswith('/') else link
            id_ = int(os.path.split(link)[1])

            available_formations[id_] = Formation(id=id_, name=name, family=family)
    
    return available_formations


def _scrape_available_playbooks(madden_year: int) -> Dict[int, Playbook]:
    '''Return available playbooks given madden year.'''
    
    playbooks_dir_url = _playbooks_dir_url(madden_year)
    response = requests.get(playbooks_dir_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    available_playbooks = {}
    for playbooks_list in soup.find_all(class_='playbooks-list'):
        type_str = playbooks_list.find_previous_sibling('h3').text
        type_ = PlaybookType[type_str.upper().replace(' ', '_')]

        for pb_element in playbooks_list.find_all(class_='playbooks-list__item'):
            name = pb_element.text.replace('\n', '')
            link = pb_element.find('a')['href']
            link = link[:-1] if link.endswith('/') else link
            id_ = int(os.path.split(link)[1])
            available_playbooks[id_] = Playbook(
                id=id_,
                name=name,
                type=type_,
                madden_year=madden_year
            )

    return available_playbooks
    

def scrape_playbook(
    madden_year: int,
    playbook_id: int,
    formation_id: int
) -> Playbook:
    '''Return specificed playbook scraped on given formations.'''

    available_playbooks = _scrape_available_playbooks(madden_year)
    assert playbook_id in available_playbooks, f'playbook_id {playbook_id} not found for madden year {madden_year}'
    playbook = available_playbooks[playbook_id]

    available_formations = _scrape_available_formations(playbook=playbook)
    assert formation_id in available_formations, f'formation_id {formation_id} not found in playbook {playbook_id}'
    formation = available_formations[formation_id]

    playbook.plays = _scrape_formation_plays(playbook=playbook, formation=formation)
    
    return playbook
