import sys

import yaml
from easydict import EasyDict as edict

import _init_paths
if __name__ == '__main__':

    if len(sys.argv)==1:
        print 'default'
    else:
        with open(sys.argv[1], 'r') as f:
            yaml_cfg = edict(yaml.load(f))
        try:
            print(yaml_cfg.TRAIN.SNAPSHOT_INFIX)
        except:
            pass
