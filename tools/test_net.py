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

""" This module is used for testing Face-MagNet models. """
from __future__ import print_function

from pip.utils import ensure_dir

import _init_paths

import argparse
import os
import pprint
import sys

import caffe

from datasets.factory import get_imdb
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_snapshot_prefix
from fast_rcnn.config import get_output_dir
from fast_rcnn.test import test_net


def parse_args():
    parser = argparse.ArgumentParser(description='Tests start here')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--net', dest='caffemodel',
                        help='.caffemodel model to load for testing',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--def', dest='prototxt',
                        help='Test prototxt file path.',
                        default='wider', type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test [wider|wider_test|fddb|pascal]',
                        default='wider', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='Visualize detections and save them to files',
                        action='store_true')
    parser.add_argument('--single', dest='single', help='Single image testing, if on --imdb should be set to the path of the image',
                        action='store_true')
    parser.add_argument('--pyramid', dest='pyramid', help='Image pyramid testing with scales [0.5,1,2]',
                        action='store_true')
    parser.add_argument('--postfix', dest='postfix', help='Experiment postfix',
                        default=None)
    parser.add_argument('--fddb_pascal_path', dest='fddb_pascal_path', help='Path to FDDB or PASCAL dataset, if not set ./dataset/{imdb}/ will be used.',
                        default=None)
    parser.add_argument('--matlab_eval', dest='matlab_eval', help='Do the matlab evaluation for WIDER',
                        action='store_true')
    parser.add_argument('--output_dir', dest='output_dir', help='Just prints out the output directory for detection results. Useful for pyramid testing',
                        action='store_true')
    parser.add_argument('--max_size', dest='max_size',
                        help='max number of detections per image',
                        default=-1, type=int)
    parser.add_argument('--min_size', dest='min_size',
                        help='min size',
                        default=-1, type=int)
    parser.add_argument('--orig_scale', dest='orig_scale',
                        help='number of bbox regression passes',
                        default=-1, type=float)
    parser.add_argument('--shuffle', dest='shuffle', help='number of bbox regression passes', default=False, action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


import os.path as osp
import csv


def gather_results_csv():
    '''
    Gathers test results of WIDER for each tested model.
    '''
    this_dir = osp.dirname(__file__)
    matlab_results_dirs = osp.join(this_dir, '..', 'matlab', 'eval_tools', 'plot', 'baselines', 'Val', 'setting_int_final')
    method_dirs = os.listdir(matlab_results_dirs)
    results_dirs = osp.join(this_dir, '..', 'output', 'results')
    if not os.path.exists(results_dirs):
        os.makedirs(results_dirs)

    settings = ['easy', 'medium', 'hard']

    with open(osp.join(results_dirs, 'all_maps_final.csv'), 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['method', 'easy', 'medium', 'hard'])
        base_name = 'wider_ap_{}_{}_val.txt'
        for dir in method_dirs:
            row = [dir]
            method_base = osp.join(matlab_results_dirs, dir, base_name)
            for setting in settings:
                with open(method_base.format(dir, setting)) as fmap:
                    row.append(fmap.readline().strip())
            csv_writer.writerow(row)


if __name__ == '__main__':
    args = parse_args()

    if args.orig_scale > 0:
        cfg.TEST.ORIG_SIZE = True
        cfg.TEST.ORIG_SIZE_SCALE = args.orig_scale

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
        args.postfix = os.path.basename(args.cfg_file)[:-4]
        print('PREFIX IS: {}'.format(args.postfix))
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    net_name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    if 'wider' in args.imdb_name:
        imdb = get_imdb(args.imdb_name)
        if not cfg.TEST.HAS_RPN:
            imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
    else:
        imdb = args.imdb_name
    postfix = args.postfix

    if args.pyramid:
        postfix += '_pyramid'
        cfg.TEST.PYRAMID = True
    if args.max_size != -1:
        cfg.TEST.MAX_SIZE = args.max_size
        postfix += '_max{}'.format(args.max_size)
    if args.min_size != -1:
        cfg.TEST.SCALES = [args.min_size, ]
        postfix += '_min{}'.format(args.min_size)
    if args.orig_scale > 0:
        postfix += '_orig_{}'.format(int(args.orig_scale * 10))

    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = net_name
    postfix = net_name + '_' + postfix
    if args.single:
        output_dir = './debug/single/'
        ensure_dir(output_dir)
    else:
        output_dir = get_output_dir(imdb, postfix=postfix)

    test_net(net, imdb, vis=args.vis, matlab_val=args.matlab_eval,
             postfix=postfix, single_img=args.single, shuffle=args.shuffle,
             fddb_pascal_path=args.fddb_pascal_path)
    if not args.shuffle:
        gather_results_csv()
