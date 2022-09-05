'''
Various utils.
'''

import time


def elapsed_ms(start_time: float) -> int:
    '''Return ms elapsed since passed start time (use time.time()).'''
    return round((time.time() - start_time)*1000)


def clamp(input, _min=0, _max=1):
    '''Clamps input between min and max.
    
    This really isn't built in to python?
    '''
    return max(_min, min(_max, input))
