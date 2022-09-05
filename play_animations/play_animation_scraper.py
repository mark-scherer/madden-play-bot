'''
Functions for scraping & saving a play animation.
'''

import sys
from os import path
from urllib.parse import urlparse
import pathlib
import subprocess
sys.path.append(path.join(path.dirname(__file__), '..')) # upwards relative imports are hacky

import glog

import constants
from play_animation import PlayAnimation

TWITTER_NETLOC = 'twitter.com'
FFMPEG_QUALITY = 1 # 1-31, lower better

def run_youtube_dl(url: str, output_path: str) -> None:
    '''Actually download media from given url to output path.'''
    command = f'youtube-dl -o "{output_path}" "{url}"'
    try:
        subprocess.run(command, check=True, shell=True)
        glog.info(f'..successfully downloaded {url} to {output_path}')
    except Exception as e:
        raise RuntimeError(f'error running youtube-dl on url: {url}: {e}') from e

def run_ffmpeg(media_path: str, output_frame_path: str) -> None:
    '''Split out single frame to process from downloaded video.'''
    command = f'ffmpeg -i {media_path} -vf "select=eq(n\,0)" -q:v {FFMPEG_QUALITY} {output_frame_path}'
    try:
        subprocess.run(command, check=True, shell=True)
        glog.info(f'..successfully split first frame out from {media_path} as {output_frame_path}')
    except Exception as e:
        raise RuntimeError(f'error running ffmpeg on video: {media_path}: {e}') from e

def scrape(url: str) -> PlayAnimation:
    '''Scrape play animation and parse first frame from given url.'''

    parsed_url = urlparse(url)
    assert parsed_url.netloc == TWITTER_NETLOC, \
        f'only supports twitter urls, found {parsed_url.netloc}'

    parsed_path = parsed_url.path
    if parsed_path[-1] == '/':
        parsed_path = parsed_path[0:-1]
    media_id = path.basename(parsed_path)
    
    media_dir = path.join(constants.PLAY_ANIMATIONS_DIR, media_id)
    media_path = path.join(media_dir, constants.MEDIA_FILENAME)
    frame_path = path.join(media_dir, constants.FRAME_FILENAME)

    pathlib.Path(constants.SCRATCH_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(media_dir).mkdir(parents=True, exist_ok=True)

    if not path.isfile(media_path):
        run_youtube_dl(url=url, output_path=media_path)
    else:
        glog.info(f'found cached media file at {media_path}, skipping download')

    if not path.isfile(frame_path):
        run_ffmpeg(media_path=media_path, output_frame_path=frame_path)
    else:
        glog.info(f'found cached frame file at {frame_path}, skipping video split')

    return PlayAnimation(
        url=url,
        media_dir=media_dir
    )

    





    