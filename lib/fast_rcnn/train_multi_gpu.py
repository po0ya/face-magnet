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

"""Multi gpu training core."""

import matplotlib

from fast_rcnn.train import SolverWrapper

matplotlib.use('Agg')

from fast_rcnn.config import cfg
import caffe
import numpy as np
from multiprocessing import Process

def solve(proto, roidb, pretrained_model, gpus, uid, rank, output_dir, max_iter,
          reload):
    caffe.set_device(gpus[rank])
    caffe.set_mode_gpu()
    caffe.set_solver_count(len(gpus))

    caffe.set_solver_rank(rank)

    caffe.set_multiprocess(True)
    cfg.GPU_ID = gpus[rank]

    solverW = SolverWrapper(solver_prototxt=proto, roidb=roidb,
                            output_dir=output_dir, gpu_id=rank,
                            pretrained_model=pretrained_model, reload=reload)
    solver = solverW.get_solver()
    nccl = caffe.NCCL(solver, uid)
    nccl.bcast()
    solver.add_callback(nccl)

    if solver.param.layer_wise_reduce:
        solver.net.after_backward(nccl)
    while solver.iter < max_iter:
        solver.step(1)

        if solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0 and rank == 0:
            solverW.snapshot()


def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb


def train_net_multi_gpu(solver_prototxt, roidb, output_dir, pretrained_model,
                        max_iter, gpus, reload):
    """Train a Fast R-CNN network."""
    roidb = filter_roidb(roidb)
    uid = caffe.NCCL.new_uid()
    caffe.init_log()
    caffe.log('Using devices %s' % str(gpus))
    procs = []

    for rank in range(len(gpus)):
        p = Process(target=solve,
                    args=(
                        solver_prototxt, roidb, pretrained_model, gpus, uid,
                        rank,
                        output_dir, max_iter, reload))
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

