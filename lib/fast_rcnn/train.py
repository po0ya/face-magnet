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

"""Modified train.py of Faster-RCNN for single gpu training."""

import _init_paths

import numpy as np
import matplotlib

matplotlib.use('Agg')

import caffe

import google.protobuf.text_format as text_format

from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
import os

from caffe.proto import caffe_pb2
import google.protobuf as pb2


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, roidb, output_dir, gpu_id,
                 pretrained_model=None, reload=False):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir
        self.gpu_id = gpu_id

        if (cfg.TRAIN.HAS_RPN and cfg.TRAIN.BBOX_REG and
                cfg.TRAIN.BBOX_NORMALIZE_TARGETS):
            # RPN can only use precomputed normalization because there are no
            # fixed statistics to compute a priori
            assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED

        if cfg.TRAIN.BBOX_REG:
            print 'Computing bounding-box regression targets...'
            self.bbox_means, self.bbox_stds = \
                rdl_roidb.add_bbox_regression_targets(roidb)
            print 'done'

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            try:
                pb2.text_format.Merge(f.read(), self.solver_param)
            except:
                text_format.Merge(f.read(), self.solver_param)

        # Copying weights from layers if asked by user in prototxt....
        ## For copying weights of "source" into a layer you should name the layer "copy_from_source"
        layer_names = self.solver.net._layer_names
        for i in xrange(len(layer_names)):
            cur_name = layer_names[i]
            if 'copy_from_' in cur_name and not 'relu' in cur_name and not reload:
                str_splits = cur_name.split('_')
                source_name = '_'.join(str_splits[2:])
                found = False
                for j in xrange(len(layer_names)):
                    if layer_names[j] == source_name:
                        found = True
                        print(
                            'Copying weight from {} to {}...\n'.format(
                                source_name,
                                cur_name))
                        for p in xrange(len(self.solver.net.params[cur_name])):
                            self.solver.net.params[cur_name][p].data[...] = \
                                self.solver.net.params[source_name][p].data
                        break
                        # if not found:
                        #     raise NameError('The source layer {} was not found for layer {}!\n'.format(source_name, cur_name))

        self.solver.net.layers[0].set_roidb(roidb, gpu_id)
        net = self.solver.net
        if reload:
            found = False
            bbox_preds = []
            for k in net.params.keys():
                if 'bbox_pred' in k and not 'rpn' in k:
                    bbox_preds.append(k)

            for bbox_pred in bbox_preds:
                print('!!! Renormalizing the final layers back !!!')
                print 'Meta file does not exist, multiplying 1/std and subtracting mean'
                net.params[bbox_pred][0].data[4:, :] = \
                    (net.params[bbox_pred][0].data[4:, :] *
                     1.0 / self.bbox_stds[4:, np.newaxis])
                net.params[bbox_pred][1].data[4:] = \
                    (net.params[bbox_pred][1].data - self.bbox_means)[
                    4:] * 1.0 / self.bbox_stds[4:]

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        scale_bbox_params = (cfg.TRAIN.BBOX_REG and
                             cfg.TRAIN.BBOX_NORMALIZE_TARGETS)
        unnormed = {}
        if scale_bbox_params:
            for k in net.params.keys():
                # Normalizing bbox_pred layers that are not in RPN
                if 'bbox_pred' in k and not 'rpn' in k:
                    bbox_pred = k
                else:
                    continue

                # save original values
                orig_0 = net.params[bbox_pred][0].data.copy()
                orig_1 = net.params[bbox_pred][1].data.copy()

                unnormed[bbox_pred] = (orig_0, orig_1)

                # scale and shift with bbox reg unnormalization; then save snapshot
                net.params[bbox_pred][0].data[...] = \
                    (net.params[bbox_pred][0].data *
                     self.bbox_stds[:, np.newaxis])
                net.params[bbox_pred][1].data[...] = \
                    (net.params[bbox_pred][1].data *
                     self.bbox_stds + self.bbox_means)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        for bbox_pred, origs in unnormed.items():
            # restore net to original state
            net.params[bbox_pred][0].data[...] = origs[0]
            net.params[bbox_pred][1].data[...] = origs[1]

        return filename

    def get_solver(self):
        return self.solver

    def train_model(self, max_iters):

        while self.solver.iter < max_iters:
            self.solver.step(1)
            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                self.snapshot()


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""

    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    if cfg.TRAIN.RANDOM_CROP_SCALE:
        print 'Appending pyramid cropped images...'
        imdb.append_cropped_scale_images()

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb


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


def train_net(solver_prototxt, roidb, output_dir,
              pretrained_model=None, max_iters=40000, reload=False, gpu_id=0):

    roidb = filter_roidb(roidb)
    sw = SolverWrapper(solver_prototxt, roidb, output_dir, gpu_id,
                       pretrained_model=pretrained_model, reload=reload)

    print 'Solving...'
    model_paths = sw.train_model(max_iters)
    sw.snapshot()
    print 'done solving'
    return model_paths
