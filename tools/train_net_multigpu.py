# Copyright 2018 The Face-MagNet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""This module contains the code for training Face-MagNet on multiple GPUs."""

import _init_paths

import argparse
import pprint
import sys

import datasets.imdb
from datasets.factory import get_imdb
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from fast_rcnn.config import get_snapshot_prefix
from fast_rcnn.train import get_training_roidb
from fast_rcnn.train_multi_gpu import train_net_multi_gpu


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Face-Magnet network')
    parser.add_argument('--gpus', dest='gpu_id',
                        help='GPUs device id to use [0]',
                        default='0,1', type=str)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=38000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='wider', type=str)
    parser.add_argument('--max_size', dest='max_size',
                        help='dataset to max size on',
                        default='10', type=str)
    parser.add_argument('--min_size', dest='max_size',
                        help='dataset to max size on',
                        default='10', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--reload', dest='reload',
                        help='Reloading saved weights. Set it if not initializing with imagenet weights.',
                        action='store_true')
    parser.add_argument('--randomize', dest='randomize',
                        help='Randomize the training.',
                        action='store_true')
    parser.add_argument('--shuffle', dest='shuffle',
                        help='Shuffle the testing order, for parallel testing',
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def combined_roidb(imdb_names):
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    gpu_id = args.gpu_id
    gpu_list = gpu_id.split(',')
    gpus = [int(i) for i in gpu_list]
    print('Using config:')
    pprint.pprint(cfg)

    imdb, roidb = combined_roidb(args.imdb_name)
    print '{:d} roidb entries'.format(len(roidb))
    snap_pre = get_snapshot_prefix(args.solver)
    output_dir = get_output_dir(imdb, postfix=snap_pre)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    train_net_multi_gpu(args.solver, roidb, output_dir,
                        pretrained_model=args.pretrained_model,
                        max_iter=args.max_iters, reload=args.reload, gpus=gpus)
