'''
Play animation dataclasses.
'''

from dataclasses import dataclass

@dataclass
class PlayAnimation:
    url: str
    media_dir: str