'''
Play animation dataclasses.
'''

from dataclasses import dataclass

@dataclass
class PlayAnimation:
    url: str
    id: str
    dir: str