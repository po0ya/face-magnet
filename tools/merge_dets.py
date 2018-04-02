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

"""Merges detection outputs of Face-MagNet. It is used to merge pyramid scale
detections that are calculated separately."""

import _init_paths

import cPickle
import os
import subprocess
import sys

import numpy as np
from pip.utils import ensure_dir

from datasets.factory import get_imdb
from fast_rcnn.nms_wrapper import nms
from test_net import gather_results_csv

if __name__ == '__main__':

    num_models = len(sys.argv) - 1
    dets = []
    all_dets = None
    output_dir = sys.argv[1]
    if len(sys.argv) == 2:
        t = os.listdir(output_dir)
        pkl_paths = [os.path.join(output_dir, p) for p in
                     filter(lambda x: 'pkl' in x, t)]
    else:
        pkl_paths = sys.argv[2:]

    fname = output_dir.split('/')[-1]
    for det_path in pkl_paths:
        assert os.path.exists(det_path)
        fname += '_' + os.path.basename(det_path).replace('.pkl', '')
        with open(det_path) as f:
            dets = cPickle.load(f)
        num_images = len(dets[1])
        if all_dets is None:
            all_dets = [[] for _ in range(num_images)]
        for i in range(num_images):
            if len(dets[1][i]) > 0:
                all_dets[i].append(dets[1][i])

    for i in range(num_images):
        if len(all_dets[i]) == 0:
            continue
        all_dets[i] = np.concatenate(all_dets[i])

    output_dir = os.path.join(output_dir, fname)
    ensure_dir(output_dir)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(2)]

    #        print impath
    for i in xrange(num_images):
        if len(all_dets[i]) == 0:
            continue

        cls_dets = all_dets[i]
        keep = nms(cls_dets.astype(np.float32), 0.3)

        cls_dets = cls_dets[keep, :]
        all_boxes[1][i] = cls_dets

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    det_all_file = os.path.join(output_dir, 'detections_all.pkl')
    with open(det_all_file, 'wb') as f:
        cPickle.dump(all_dets, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb = get_imdb('wider_val')
    imdb.evaluate_detections(all_boxes, output_dir)

    path = os.path.join(os.path.dirname(__file__),
                        '..', 'matlab', 'eval_tools')
    cmd = 'cd {} && '.format(path)
    cmd += 'matlab -nodisplay -nodesktop '
    cmd += '-r "wider_eval(\'{:s}\',\'{:s}\'); quit;"' \
        .format(output_dir, fname)
    print('Running:\n{}'.format(cmd))
    status = subprocess.call(cmd, shell=True)
    gather_results_csv()
