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

"""Modified test.py of Faster-RCNN."""

import _init_paths

import cPickle
import os
import subprocess

import cv2
import numpy as np
from matplotlib import pyplot as plt

from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.nms_wrapper import nms
from utils.blob import im_list_to_blob
from utils.timer import Timer


def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    pixel_means = cfg.PIXEL_MEANS
    im_orig -= pixel_means

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        if cfg.TEST.ORIG_SIZE:
            im_scale = cfg.TEST.ORIG_SIZE_SCALE
        else:
            im_scale = float(target_size) / float(im_size_min)
            # im_scale = np.sqrt((cfg.TEST.MAX_SIZE * cfg.TEST.MAX_SIZE) * 1.0 / (im_size_max * im_size_min))

        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)

        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels


def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data': None, 'rois': None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors


def im_detect(net, im, boxes=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
        bb = pred_boxes[:, 4:]

    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes


def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes


def vis_boxes_single(im, bboxes, plt_name='vis_box.jpg', out_dir='./debug/vis/'):
    """Draw detected bounding boxes."""
    inds = np.where(bboxes[:, -1] >= 0.7)[0]
    bboxes = bboxes[inds]
    fig, ax = plt.subplots(figsize=(12, 12))

    if im.shape[0] == 3:
        im_cp = im.copy()
        im_cp = im_cp.transpose((1, 2, 0))
        if im.min() < 0:
            pixel_means = cfg.PIXEL_MEANS
            im_cp = im_cp + pixel_means
        im = im_cp.astype(dtype=np.uint8)

    im = im[:, :, (2, 1, 0)]
    offset = [0, 0]
    dims = im.shape[:1]
    ax.imshow(im, aspect='equal')
    if bboxes.shape[0] != 0:

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, :]
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor=(0, 2 * bbox[4] - 1, 2 - 2 * bbox[4]), linewidth=2)
            )

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(os.path.join(out_dir, plt_name), bbox_inches='tight')
    print('saved {}'.format(os.path.join(out_dir, plt_name)))
    plt.clf()
    plt.cla()


def test_net(net, imdb, thresh=0.05, vis=False, matlab_val=False, postfix='', single_img=False,
             shuffle=False, fddb_pascal_path='./data/fddb/'):
    """ Test a network on an image set.

    Args:
        net: Caffe net
        imdb: Either an imdb object, or a string [fddb|pascal], or if single_img = True; a string which is the path to a single image
        thresh: Score threshold.
        matlab_eval: For wider dataset, it does matlab evaluation if this flag is on.
        postfix: A postfix for saving the result.
        shuffle: Shuffles the testing order, for parallel testing, just works on WIDER.
        fddb_pascal_path: Path to Pascal-Faces or FDDB images.
    """
    _t = {'im_detect': Timer(), 'misc': Timer()}

    vis_single = False

    def get_image_boxes(impath):
        '''
        Auxilary function to get the detection results for one image
        Arguments:
            impath: string path to the image
        '''
        im = cv2.imread(impath)
        imfname = os.path.basename(impath)
        _t['im_detect'].tic()
        if not cfg.TEST.PYRAMID:
            scores, boxes = im_detect(net, im)
            boxes = boxes[:, 4:8]
        else:
            all_scores = []
            all_boxes = []
            cfg.TEST.ORIG_SIZE = True
            for s in cfg.TEST.PYRAMID_SCALES:
                im_cp = cv2.resize(im, (0, 0), fx=s, fy=s).astype(
                    np.float32, copy=True)
                scores, boxes = im_detect(net, im_cp)
                boxes = boxes[:, 4:8]
                r = (1.0 / s)
                all_scores.append(np.copy(scores))
                all_boxes.append(boxes * r)

            scores = np.concatenate(all_scores)
            boxes = np.concatenate(all_boxes)

        _t['im_detect'].toc()

        _t['misc'].tic()
        # skip j = 0, because it's the background class
        for j in xrange(1, 2):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, :]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            cls_dets_all = np.copy(cls_dets)
            keep = nms(cls_dets, cfg.TEST.NMS)

            cls_dets = cls_dets[keep, :]
            if vis or vis_single:
                vis_boxes_single(im, cls_dets, plt_name=imfname.replace('.jpg', '_{}_detections.pdf'.format(postfix)))
            _t['misc'].toc()
            return cls_dets

    if isinstance(imdb, basestring):
        assert imdb == 'fddb' or imdb == 'pascal' or single_img

        if single_img:
            vis_single = True
            return get_image_boxes(imdb)
        if not fddb_pascal_path:
            fddb_pascal_path = './data/{}/'.format(imdb)

        results_dir = './output/results/'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        results_filepath = os.path.join(results_dir, '{}_evals_{}_{}.txt'.format(imdb, net.name, postfix))
        if os.path.exists(results_filepath):
            print 'detection results {} exists'.format(results_filepath)

        dbname = imdb
        result_lines = []
        if dbname == 'pascal':
            img_list_file = os.path.join(fddb_pascal_path, 'imglist.csv')
            with open(img_list_file, 'r') as f:
                img_list = f.readlines()
                num_images = len(img_list)
                i = 0

                for img_path in img_list:
                    img_path = img_path.strip()
                    img_name = os.path.basename(img_path)

                    img_bboxes = get_image_boxes(img_path)
                    for det in img_bboxes:
                        result_lines.append(
                            '{} {} {:d} {:d} {:d} {:d}\n'.format(
                                img_name, det[4], int(det[0]), int(det[1]), int(det[2]), int(det[3])))

                    print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
                        .format(i + 1, num_images, _t['im_detect'].average_time,
                                _t['misc'].average_time)
                    i = i + 1

        elif dbname == 'fddb':
            annot_files = os.path.join('./data/fddb/FDDB-folds/allEllipseList.txt')
            with open(annot_files, 'r') as f:
                annot_lines = [l.strip() for l in f.readlines()]
                num_images = len(annot_lines)
                ctr = 0
                while ctr != len(annot_lines):
                    im_name = annot_lines[ctr]
                    abs_im_path = os.path.join(fddb_pascal_path, 'originalPics', im_name + '.jpg')
                    result_lines.append(im_name + '\n')
                    img_bboxes = get_image_boxes(abs_im_path)

                    if img_bboxes is not None:
                        result_lines.append(str(img_bboxes.shape[0]) + '\n')
                        for det in img_bboxes:
                            result_lines.append('{:d} {:d} {:d} {:d} {} \n'.format(
                                int(det[0]), int(det[1]), int(det[2]) - int(det[0]), int(det[3]) - int(det[1]), det[4]))
                    else:
                        print('######### WARNING ############ NO BOXES {}'.format(abs_im_path))
                        result_lines.append('0\n')

                    if ctr + 1 != len(annot_lines):
                        n = int(annot_lines[ctr + 1])
                        ctr = ctr + n + 2

                    print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
                        .format(ctr + 1, num_images, _t['im_detect'].average_time,
                                _t['misc'].average_time)

        if result_lines is not None:
            print 'writing to ', results_filepath
            with open(results_filepath, 'w') as f:
                f.writelines(result_lines)

        return

    # If not we're testing on WIDER

    num_images = len(imdb.image_index)

    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    all_dets = [[] for _ in xrange(num_images)]
    rois = [[] for _ in xrange(num_images)]

    output_dir = get_output_dir(imdb, postfix=postfix)
    print('Output dir: {}'.format(output_dir))
    det_file = os.path.join(output_dir, 'detections.pkl')

    # timers
    inds = range(num_images)
    if shuffle:
        np.random.shuffle(inds)

    for i in inds:
        # filter out any ground truth boxes
        impath = imdb.image_path_at(i)

        try:
            all_boxes[1][i] = imdb.get_bbox(i, output_dir)
        except:
            print('[#] Could not load detected boxes for {}'.format(impath))
            all_boxes[1][i] = get_image_boxes(impath)
            imdb.evaluate_single_detection(i, all_boxes[1][i], output_dir)

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
            .format(i + 1, num_images, _t['im_detect'].average_time,
                    _t['misc'].average_time)
    if not shuffle:
        det_file = os.path.join(output_dir, 'detections.pkl')
        with open(det_file, 'wb') as f:
            cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'

    if postfix is not None:
        net.name = net.name + '_' + cfg.EXP_DIR + '_' + postfix
    if matlab_val:
        path = os.path.join(os.path.dirname(__file__),
                            '..', '..', 'matlab', 'eval_tools')
        cmd = 'cd {} && '.format(path)
        cmd += 'matlab -nodisplay -nodesktop '
        cmd += '-r "dbstop if error; '
        cmd += 'wider_eval(\'{:s}\',\'{:s}\'); quit;"' \
            .format(output_dir, net.name)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)
