"""Factory method for easily getting imdbs by name."""
from datasets.wider import wider

import numpy as np
__sets = {}
for split in ['train','val','test','train_val']:
    name = 'wider_{}'.format(split)
    __sets[name] = (lambda split=split: wider(split,wider_path='./data/wider'))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
