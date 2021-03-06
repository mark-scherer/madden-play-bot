'''
Script for scraping and parsing play animations.
'''

import glog

import play_animation_scraper
import play_animation_parser

TEST_PLAY_ANIMATION_URL = 'https://twitter.com/NextGenStats/status/1493714866200199172'
# TEST_PLAY_ANIMATION_URL = 'https://twitter.com/NextGenStats/status/1485447699600003077'

def main():
    url = TEST_PLAY_ANIMATION_URL

    # scrape play animation
    scraped_play_animation = play_animation_scraper.scrape(url=url)
    parsed_play_animation = play_animation_parser.parse(scraped_play_animation=scraped_play_animation)
    glog.info(f'finished scraping & parsing play animation from url: {url}: {parsed_play_animation}')

main()
