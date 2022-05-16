'''
Functions for scraping entire madden playbook from hustle.gg.
'''

from typing import List, Dict, NamedTuple
import os
import json

import requests
import glog
from bs4 import BeautifulSoup

from playbook import Play, Formation, Playbook, PlaybookType, FormationFamily

SUPPORTED_YEARS = [21, 22]

class PlayScrapingInfo(NamedTuple):
    name: str
    image_url: str


def _playbooks_dir_url(madden_year: int) -> str:
    assert madden_year in SUPPORTED_YEARS, f'madden_year {madden_year} not supported - only supports {SUPPORTED_YEARS}'
    return f'https://huddle.gg/{madden_year}/playbooks/'

def _playbook_url(madden_year: int, playbook_id: int) -> str:
    return os.path.join(_playbooks_dir_url(madden_year), str(playbook_id))

def _formation_url(madden_year: int, playbook_id: int, formation_id: str) -> str:
    return os.path.join(_playbook_url(madden_year, playbook_id), 'formations', str(formation_id))

def _scrape_formation_plays(playbook: Playbook, formation: Formation) -> List[Play]:
    '''Return list of scraped plays given playbook and formation.'''
    formation_url = _formation_url(
        madden_year=playbook.madden_year,
        playbook_id=playbook.id,
        formation_id=formation.id
    )
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

def _scrape_playbook_formations(playbook: Playbook) -> List[Formation]:
    '''Return list of scraped formations given playbook.'''
    playbook_url = _playbook_url(playbook.madden_year, playbook.id)
    response = requests.get(playbook_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    
    formations = []
    for formations_list in soup.find_all(class_='playbooks-list'):
        family_str = formations_list.find_previous_sibling('h3').text
        family = FormationFamily[family_str.upper().replace(' ', '_')]
        for formation_element in formations_list.find_all(class_='playbooks-list__item'):
            name = formation_element.text.replace('\n', '')
            link = formation_element.find('a')['href']
            link = link[:-1] if link.endswith('/') else link
            id_ = int(os.path.split(link)[1])
            formations.append(Formation(id=id_, name=name, family=family))
    return formations

def _scrape_playbook_plays(playbook: Playbook) -> List[Play]:
    '''Return list of scraped plays given playbook.'''
    formations = _scrape_playbook_formations(playbook)
    # glog.info(f'scraped {len(formations)} formations: {formations}')
    plays = []
    for formation in formations[0:2]:
        # glog.info(f'scraping formation\'s plays: {formation}')
        plays += _scrape_formation_plays(playbook, formation)
    return plays

def _scrape_available_playbooks(madden_year: int) -> List[Playbook]:
    '''Return list of available playbooks given madden year.'''
    playbooks_dir_url = _playbooks_dir_url(madden_year)
    response = requests.get(playbooks_dir_url)
    response.raise_for_status()

    # DEBUG
    # response_file = os.path.join('/Users/mark/Downloads', 'huddle_gg_playbooks_dir_response.html')
    # with open(response_file, 'w') as f:
    #     f.write(response.text)
    # glog.info(f'saved playbooks dir response to {response_file}')

    soup = BeautifulSoup(response.text, 'html.parser')
    playbooks = []
    for playbooks_list in soup.find_all(class_='playbooks-list'):
        type_str = playbooks_list.find_previous_sibling('h3').text
        type_ = PlaybookType[type_str.upper().replace(' ', '_')]
        for pb_element in playbooks_list.find_all(class_='playbooks-list__item'):
            name = pb_element.text.replace('\n', '')
            link = pb_element.find('a')['href']
            link = link[:-1] if link.endswith('/') else link
            id_ = int(os.path.split(link)[1])
            playbooks.append(Playbook(
                id=id_,
                name=name,
                type=type_,
                madden_year=madden_year
            ))
    return playbooks

def scrape_madden_year_playbooks(madden_year: int) -> List[Playbook]:
    '''Return list of scraped playbooks populated with plays for given madden year.
    Note: only returns offensive playbooks
    '''
    available_playbooks = _scrape_available_playbooks(madden_year)
    scraped_playbooks = []
    for playbook in available_playbooks[0:1]:
        if playbook.type != PlaybookType.OFFENSE:
            continue
        # glog.info(f'scraping offensive playbook: {playbook}')
        playbook.plays = _scrape_playbook_plays(playbook)
        scraped_playbooks.append(playbook)
    return scraped_playbooks

def scrape_playbooks(madden_years: List[int]) -> List[Playbook]:
    '''Return list of all scraped playbooks for provided madden years.'''
    playbooks = []
    for year in madden_years:
        playbooks += scrape_madden_year_playbooks(year)
    return playbooks
