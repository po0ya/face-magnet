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

"""Similar to `lib/datasets/imdb.py` of https://github.com/rbgirshick/py-faster-rcnn with more data augmentations."""

import os
import os.path as osp
import PIL
from utils.cython_bbox import bbox_overlaps
import numpy as np
import scipy.sparse
from fast_rcnn.config import cfg


class imdb(object):
    """Image database."""

    def __init__(self, name):
        self._name = name
        self._num_classes = 0
        self._classes = []
        self._image_index = []
        self._obj_proposer = 'selective_search'
        self._roidb = None
        self._roidb_handler = self.default_roidb
        # Use this dict for storing dataset specific config options
        self.config = {}

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    def set_proposal_method(self, method):
        method = eval('self.' + method + '_roidb')
        self.roidb_handler = method

    @property
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()
        return self._roidb

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def num_images(self):
        return len(self.image_index)

    def image_path_at(self, i):
        raise NotImplementedError

    def default_roidb(self):
        raise NotImplementedError

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError

    def _get_widths(self):
        try:
            return [self.roidb[i]['image_size'][0]
                    for i in xrange(self.num_images)]
        except:
            for i in xrange(self.num_images):
                size = PIL.Image.open(self.image_path_at(i)).size
                self.roidb[i]['image_size'] = size

            return [self.roidb[i]['image_size'][0]
                    for i in xrange(self.num_images)]


    def append_blurred_images(self):
        num_images = self.num_images
        for i in xrange(num_images):
            self.roidb[i]['blurred'] = False
            if cfg.TRAIN.USE_BLURRED:
                entry = {k: v for (k, v) in self.roidb[i].items()}
                entry['blurred'] = True
                self.roidb.append(entry)
        if cfg.TRAIN.USE_BLURRED:
            self._image_index = self._image_index * 2

    def append_scaled_images(self):
        num_images = self.num_images
        for k in cfg.TRAIN.ORIG_SCALE_MULTS:
            for i in xrange(num_images):
                if k != 1:
                    entry = {k: v for (k, v) in self.roidb[i].items()}
                    entry['scale'] = k
                    boxes = np.floor(self.roidb[i]['boxes'].copy().astype(np.float32) * k).astype(np.int32)
                    entry['boxes'] = boxes
                    self.roidb.append(entry)
                else:
                    self.roidb[i]['scale'] = k

        self._image_index = self._image_index * len(cfg.TRAIN.ORIG_SCALE_MULTS)

    def append_flipped_images(self):
        num_images = self.num_images
        widths = self._get_widths()
        for i in xrange(num_images):
            boxes = self.roidb[i]['boxes'].copy()

            # crop_box = np.array(self.roidb[i]['crop_box']).copy()
            # crop_box[0] = widths[i] - self.roidb[i]['crop_box'][2] - 1
            # crop_box[2] = widths[i] - self.roidb[i]['crop_box'][0] - 1

            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()

            entry = {k: v for (k, v) in self.roidb[i].items()}
            entry['flipped'] = True
            entry['boxes'] = boxes

            self.roidb.append(entry)
        self._image_index = self._image_index * 2

    def append_cropped_scale_images(self):
        def get_area(rect):
            return (rect[2] - rect[0]) * (rect[3] - rect[1])

        def _get_random_in_box(rect):
            return [np.random.randint(rect[0], high=rect[2]), np.random.randint(rect[1], high=rect[3])]

        def _get_inside_box(image_size, scale):
            r = [
                cfg.CROP_SIZE/2,
                cfg.CROP_SIZE/2,
                np.floor(image_size[0]* scale-cfg.CROP_SIZE/2.0).astype(np.int32),
                np.floor(image_size[1]* scale-cfg.CROP_SIZE/2.0).astype(np.int32),
            ]
            assert r[0] < r[2],'xs not right'
            assert r[1] < r[3],'ys not right'
            return r


        def _clip_boxes_rel(base_box, bboxes):
            inside_flag = np.zeros([bboxes.shape[0]], dtype=np.bool)
            bboxes = bboxes.copy()
            bb_c = bboxes.copy()
            for i in range(bboxes.shape[0]):
                bboxes[i, 0] = np.max((base_box[0], bboxes[i, 0])) - base_box[0]
                bboxes[i, 1] = np.max((base_box[1], bboxes[i, 1])) - base_box[1]
                bboxes[i, 2] = np.min((base_box[2], bboxes[i, 2])) - base_box[0]
                bboxes[i, 3] = np.min((base_box[3], bboxes[i, 3])) - base_box[1]
                if (bboxes[i, 0] >= 0 and bboxes[i, 1] >= 0 and bboxes[i, 2] > 0 and bboxes[i, 3] > 0 and
                             bboxes[i, 0] < base_box[2] - base_box[0] and
                             bboxes[i, 1] < base_box[3] - base_box[1] and
                             bboxes[i, 2] < base_box[2] - base_box[0] and
                             bboxes[i, 3] < base_box[3] - base_box[1]):
                        assert bboxes[i, 0] < bboxes[i,2]
                        assert bboxes[i, 1] < bboxes[i, 3]
                        inside_flag[i] = True


                    #if (get_area(bb_c[i, :]) > 0 and (get_area(bboxes[i, :]) * 1.0 / get_area(bb_c[i, :])) > 0.5):

            return bboxes[inside_flag, :], inside_flag

        num_images = self.num_images
        scales = [1,0.5,2]
        for i in xrange(num_images):
            for ss in range(6):
                entry = {k: v for (k, v) in self.roidb[i].items()}
                scale = scales[ss%3]
                w_ratio = entry['image_size'][0] * scale * 1.0 / cfg.CROP_SIZE
                h_ratio = entry['image_size'][1] * scale * 1.0 / cfg.CROP_SIZE

                if w_ratio <= 1.2 or h_ratio <= 1.2:
                    continue

                inside_box = _get_inside_box(np.array(entry['image_size']), scale)

                random_center = _get_random_in_box(inside_box)

                half_crop_w = cfg.CROP_SIZE/2
                half_crop_h = cfg.CROP_SIZE/2


                boxes = np.floor(self.roidb[i]['boxes'].copy().astype(np.float32) * scale).astype(np.int32)

                crop_box = [random_center[0] - half_crop_w, random_center[1] - half_crop_h,
                            random_center[0] + half_crop_w, random_center[1] + half_crop_h]

                [inside_boxes, inside_inds] = _clip_boxes_rel(crop_box, boxes)
                ctr = 0
                while (inside_boxes.shape[0] == 0 and ctr < 4):
                    ctr += 1
                    random_center = _get_random_in_box(inside_box)
                    crop_box = [random_center[0] - half_crop_w, random_center[1] - half_crop_h,
                                random_center[0] + half_crop_w, random_center[1] + half_crop_h]
                    [inside_boxes, inside_inds] = _clip_boxes_rel(crop_box, boxes)

                if (inside_boxes.shape == 0):
                    random_center = np.random.randint(0, boxes.shape[0] - 1)
                    center = [np.floor((boxes[random_center, 0] + boxes[random_center, 2]) / 2).astype(np.int32),
                              np.floor((boxes[random_center, 1] + boxes[random_center, 3]) / 2).astype(np.int32)]

                    crop_box = [center[0] - half_crop_w, center[1] - half_crop_h,
                                center[0] + half_crop_w, center[1] + half_crop_h]

                    if (crop_box[0] < 0):
                        crop_box[2] -= crop_box[0]
                        crop_box[0] = 0
                    if (crop_box[1] < 0):
                        crop_box[3] -= crop_box[1]
                        crop_box[1] = 0

                    if (crop_box[2] > entry['image_size'][0]):
                        crop_box[0] -= crop_box[2] - entry['image_size'][0]
                        crop_box[2] = entry['image_size'][0]

                    if (crop_box[3] > entry['image_size'][1]):
                        crop_box[1] -= crop_box[3] - entry['image_size'][1]
                        crop_box[3] = entry['image_size'][1]
                    inside_boxes, inside_inds = _clip_boxes_rel(crop_box, boxes)

                if (inside_boxes.shape == 0):
                    continue

                entry['crop_box'] = crop_box
                entry['gt_classes'] = np.array(entry['gt_classes'])[inside_inds]
                entry['boxes'] = inside_boxes
                entry['gt_overlaps'] = entry['gt_overlaps'][inside_inds]
                entry['image_size'] = [half_crop_w * 2, half_crop_h * 2]
                entry['scale'] =scale
                self.roidb.append(entry)
                self._image_index.append(self._image_index[i])

    def append_cropped_images(self):
        def get_area(rect):
            return (rect[2] - rect[0]) * (rect[3] - rect[1])

        def _get_random_in_box(rect):
            return [np.random.randint(rect[0], high=rect[2]), np.random.randint(rect[1], high=rect[3])]

        def _get_inside_box(image_size, scale):
            return [
                np.floor(image_size[0] * 0.5 / scale).astype(np.int32),
                np.floor(image_size[1] * 0.5 / scale).astype(np.int32),
                np.floor(image_size[0] - image_size[0] * 0.5 / scale).astype(np.int32),
                np.floor(image_size[1] - image_size[1] * 0.5 / scale).astype(np.int32),
            ]

            return [
                np.floor(image_size[0] * 0.5 / scale).astype(np.int32),
                np.floor(image_size[1] * 0.5 / scale).astype(np.int32),
                np.floor(image_size[0] - image_size[0] * 0.5 / scale).astype(np.int32),
                np.floor(image_size[1] - image_size[1] * 0.5 / scale).astype(np.int32),
            ]

        def _clip_boxes_rel(base_box, bboxes):
            inside_flag = np.zeros([bboxes.shape[0]], dtype=np.bool)
            bboxes = bboxes.copy()
            bb_c = bboxes.copy()
            for i in range(bboxes.shape[0]):
                bboxes[i, 0] = np.max((base_box[0], bboxes[i, 0])) - base_box[0]
                bboxes[i, 1] = np.max((base_box[1], bboxes[i, 1])) - base_box[1]
                bboxes[i, 2] = np.min((base_box[2], bboxes[i, 2])) - base_box[0]
                bboxes[i, 3] = np.min((base_box[3], bboxes[i, 3])) - base_box[1]
                if ((bboxes[i, 0] >= 0 and bboxes[i, 1] >= 0 and
                             bboxes[i, 0] < base_box[2] - base_box[0] and
                             bboxes[i, 1] < base_box[3] - base_box[1]) and
                        (bboxes[i, 2] < base_box[2] - base_box[0] and bboxes[i, 3] < base_box[3] - base_box[1])):
                    if (get_area(bb_c[i, :]) > 0 and (get_area(bboxes[i, :]) * 1.0 / get_area(bb_c[i, :])) > 0.5):
                        inside_flag[i] = True

            return bboxes[inside_flag, :], inside_flag

        num_images = self.num_images
        scales = [1,0.5,2]
        for k in scales:
            for i in xrange(num_images):
                if k != 1:
                    entry = {k: v for (k, v) in self.roidb[i].items()}
                    for ss in range(12):


                        inside_box = _get_inside_box(entry['image_size'], k)

                        random_center = _get_random_in_box(inside_box)

                        half_crop_w = np.floor(entry['image_size'][0] * 0.5 / k).astype(np.int32)
                        half_crop_h = np.floor(entry['image_size'][1] * 0.5 / k).astype(np.int32)


                        boxes = self.roidb[i]['boxes'].copy()

                        crop_box = [random_center[0] - half_crop_w, random_center[1] - half_crop_h,
                                    random_center[0] + half_crop_w, random_center[1] + half_crop_h]

                        [inside_boxes, inside_inds] = _clip_boxes_rel(crop_box, boxes)
                        ctr = 0
                        while (inside_boxes.shape[0] == 0 and ctr < 4):
                            ctr += 1
                            random_center = _get_random_in_box(inside_box)
                            crop_box = [random_center[0] - half_crop_w, random_center[1] - half_crop_h,
                                        random_center[0] + half_crop_w, random_center[1] + half_crop_h]
                            [inside_boxes, inside_inds] = _clip_boxes_rel(crop_box, boxes)

                        if (inside_boxes.shape == 0):
                            random_center = np.random.randint(0, boxes.shape[0] - 1)
                            center = [np.floor((boxes[random_center, 0] + boxes[random_center, 2]) / 2).astype(np.int32),
                                      np.floor((boxes[random_center, 1] + boxes[random_center, 3]) / 2).astype(np.int32)]

                            crop_box = [center[0] - half_crop_w, center[1] - half_crop_h,
                                        center[0] + half_crop_w, center[1] + half_crop_h]

                            if (crop_box[0] < 0):
                                crop_box[2] -= crop_box[0]
                                crop_box[0] = 0
                            if (crop_box[1] < 0):
                                crop_box[3] -= crop_box[1]
                                crop_box[1] = 0

                            if (crop_box[2] > entry['image_size'][0]):
                                crop_box[0] -= crop_box[2] - entry['image_size'][0]
                                crop_box[2] = entry['image_size'][0]

                            if (crop_box[3] > entry['image_size'][1]):
                                crop_box[1] -= crop_box[3] - entry['image_size'][1]
                                crop_box[3] = entry['image_size'][1]
                            inside_boxes, inside_inds = _clip_boxes_rel(crop_box, boxes)

                        if (inside_boxes.shape == 0):
                            continue

                        entry['crop_box'] = crop_box
                        entry['gt_classes'] = np.array(entry['gt_classes'])[inside_inds]
                        entry['boxes'] = inside_boxes
                        entry['gt_overlaps'] = entry['gt_overlaps'][inside_inds]
                        entry['image_size'] = [half_crop_w * 2, half_crop_h * 2]
                       # entry['scale'] = sca
                        self.roidb.append(entry)
                        self._image_index.append(self._image_index[i])
                else:
                    self.roidb[i]['crop_box'] = [0, 0, self.roidb[i]['image_size'][0] - 1,
                                                 self.roidb[i]['image_size'][1] - 1]

    def evaluate_recall(self, candidate_boxes=None, thresholds=None,
                        area='all', limit=None):
        """Evaluate detection proposal recall metrics.

        Returns:
            results: dictionary of results with keys
                'ar': average recall
                'recalls': vector recalls at each IoU overlap threshold
                'thresholds': vector of IoU overlap thresholds
                'gt_overlaps': vector of all ground-truth overlaps
        """
        # Record max overlap value for each gt box
        # Return vector of overlap values
        areas = {'all': 0, 'small': 1, 'medium': 2, 'large': 3,
                 '96-128': 4, '128-256': 5, '256-512': 6, '512-inf': 7}
        area_ranges = [[0 ** 2, 1e5 ** 2],  # all
                       [0 ** 2, 32 ** 2],  # small
                       [32 ** 2, 96 ** 2],  # medium
                       [96 ** 2, 1e5 ** 2],  # large
                       [96 ** 2, 128 ** 2],  # 96-128
                       [128 ** 2, 256 ** 2],  # 128-256
                       [256 ** 2, 512 ** 2],  # 256-512
                       [512 ** 2, 1e5 ** 2],  # 512-inf
                       ]
        assert areas.has_key(area), 'unknown area range: {}'.format(area)
        area_range = area_ranges[areas[area]]
        gt_overlaps = np.zeros(0)
        num_pos = 0
        for i in xrange(self.num_images):
            # Checking for max_overlaps == 1 avoids including crowd annotations
            # (...pretty hacking :/)
            max_gt_overlaps = self.roidb[i]['gt_overlaps'].toarray().max(axis=1)
            gt_inds = np.where((self.roidb[i]['gt_classes'] > 0) &
                               (max_gt_overlaps == 1))[0]
            gt_boxes = self.roidb[i]['boxes'][gt_inds, :]
            gt_areas = self.roidb[i]['seg_areas'][gt_inds]
            valid_gt_inds = np.where((gt_areas >= area_range[0]) &
                                     (gt_areas <= area_range[1]))[0]
            gt_boxes = gt_boxes[valid_gt_inds, :]
            num_pos += len(valid_gt_inds)

            if candidate_boxes is None:
                # If candidate_boxes is not supplied, the default is to use the
                # non-ground-truth boxes from this roidb
                non_gt_inds = np.where(self.roidb[i]['gt_classes'] == 0)[0]
                boxes = self.roidb[i]['boxes'][non_gt_inds, :]
            else:
                boxes = candidate_boxes[i]
            if boxes.shape[0] == 0:
                continue
            if limit is not None and boxes.shape[0] > limit:
                boxes = boxes[:limit, :]

            overlaps = bbox_overlaps(boxes.astype(np.float),
                                     gt_boxes.astype(np.float))

            _gt_overlaps = np.zeros((gt_boxes.shape[0]))
            for j in xrange(gt_boxes.shape[0]):
                # find which proposal box maximally covers each gt box
                argmax_overlaps = overlaps.argmax(axis=0)
                # and get the iou amount of coverage for each gt box
                max_overlaps = overlaps.max(axis=0)
                # find which gt box is 'best' covered (i.e. 'best' = most iou)
                gt_ind = max_overlaps.argmax()
                gt_ovr = max_overlaps.max()
                assert (gt_ovr >= 0)
                # find the proposal box that covers the best covered gt box
                box_ind = argmax_overlaps[gt_ind]
                # record the iou coverage of this gt box
                _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                assert (_gt_overlaps[j] == gt_ovr)
                # mark the proposal box and the gt box as used
                overlaps[box_ind, :] = -1
                overlaps[:, gt_ind] = -1
            # append recorded iou coverage level
            gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

        gt_overlaps = np.sort(gt_overlaps)
        if thresholds is None:
            step = 0.05
            thresholds = np.arange(0.5, 0.95 + 1e-5, step)
        recalls = np.zeros_like(thresholds)
        # compute recall for each iou threshold
        for i, t in enumerate(thresholds):
            recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
        # ar = 2 * np.trapz(recalls, thresholds)
        ar = recalls.mean()
        return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
                'gt_overlaps': gt_overlaps}

    def create_roidb_from_box_list(self, box_list, gt_roidb):
        assert len(box_list) == self.num_images, \
            'Number of boxes must match number of ground-truth images'
        roidb = []
        for i in xrange(self.num_images):
            boxes = box_list[i]
            num_boxes = boxes.shape[0]
            overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)

            if gt_roidb is not None and gt_roidb[i]['boxes'].size > 0:
                gt_boxes = gt_roidb[i]['boxes']
                gt_classes = gt_roidb[i]['gt_classes']
                gt_overlaps = bbox_overlaps(boxes.astype(np.float),
                                            gt_boxes.astype(np.float))
                argmaxes = gt_overlaps.argmax(axis=1)
                maxes = gt_overlaps.max(axis=1)
                I = np.where(maxes > 0)[0]
                overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

            overlaps = scipy.sparse.csr_matrix(overlaps)
            roidb.append({
                'boxes': boxes,
                'gt_classes': np.zeros((num_boxes,), dtype=np.int32),
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': np.zeros((num_boxes,), dtype=np.float32),
            })
        return roidb

    @staticmethod
    def merge_roidbs(a, b):
        assert len(a) == len(b)
        for i in xrange(len(a)):
            a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
            a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                            b[i]['gt_classes']))
            a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                       b[i]['gt_overlaps']])
            a[i]['seg_areas'] = np.hstack((a[i]['seg_areas'],
                                           b[i]['seg_areas']))
        return a

    def competition_mode(self, on):
        """Turn competition mode on or off."""
        pass
