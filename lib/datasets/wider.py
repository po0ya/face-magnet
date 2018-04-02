import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from fast_rcnn.config import cfg
import csv
from PIL import Image


class wider(imdb):

    def __init__(self, split, wider_path=None):
        self._test_flag = False
        self._split = split
        imdb.__init__(self, 'wider_' + split)
        self._image_set = split
        if wider_path is None:
            self._dataset_path = self._get_default_path()
        else:
            self._dataset_path = wider_path
        self._imgs_path = os.path.join(self._dataset_path, 'WIDER_{}'.format(split), 'images')
        csv_path = os.path.join(self._dataset_path, 'WIDER_{}'.format(split), 'imglist.csv')

        if not os.path.exists(csv_path) and split != 'test':
            path = os.path.join(os.path.dirname(__file__),
                                '..', '..', 'matlab')
            cmd = 'cd {} && '.format(path)
            cmd += 'matlab -nodisplay -nodesktop '
            cmd += '-r "dbstop if error; '
            cmd += 'wider_csv(\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
                .format(self._dataset_path, split,
                        csv_path)
            print('Running:\n{}'.format(cmd))
            status = subprocess.call(cmd, shell=True)
        elif split == 'test':
            self._test_flag = True

        self._fp_bbox_map = {}
        prev_file = 'blah'
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            if not self._test_flag:
                for row in reader:
                    if row[0] != prev_file:
                        self._fp_bbox_map[row[0]] = []
                    prev_file = row[0]

                    x1 = max(0, int(row[1]))
                    y1 = max(0, int(row[2]))
                    w = int(row[3])
                    h = int(row[4])
                    self._fp_bbox_map[row[0]].append([x1, y1,
                                                      x1 + w,
                                                      y1 + h])
                self._image_paths = self._fp_bbox_map.keys()
            else:
                self._image_paths = []
                for row in reader:
                    self._image_paths.append(row[0])

        self._image_index = range(len(self._image_paths))
        self._classes = ['_bg', 'face']

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'wider')

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._imgs_path, self._image_paths[index])
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """

        cache_file = os.path.join(self.cache_path, '{}_{}_gt_roidb.pkl'.format(self.name, self._split))
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        roidb = []

        for fp in self._image_paths:
            # Make pixel indexes 0-based
            if self._test_flag:
                roidb.append({'image_size': Image.open(os.path.join(self._imgs_path, fp)).size,
                              'file_path': os.path.join(self._imgs_path, fp)})
            else:
                boxes = np.zeros([len(self._fp_bbox_map[fp]), 4], np.float)

                gt_classes = np.ones([len(self._fp_bbox_map[fp])], np.int32)
                overlaps = np.zeros([len(self._fp_bbox_map[fp]), 2], np.float)

                ix = 0

                for bbox in self._fp_bbox_map[fp]:
                    imsize = Image.open(os.path.join(self._imgs_path, fp)).size

                    x1 = bbox[0]
                    y1 = bbox[1]
                    x2 = min(imsize[0], bbox[2])
                    y2 = min(imsize[1], bbox[3])

                    if (x2 - x1) < 1 or y2 - y1 < 1:
                        continue

                    boxes[ix, :] = np.array([x1, y1, x2, y2], np.float)

                    cls = int(1)
                    gt_classes[ix] = cls
                    overlaps[ix, cls] = 1.0
                    ix += 1
                overlaps = scipy.sparse.csr_matrix(overlaps)

                roidb.append({'boxes': boxes,
                              'gt_classes': gt_classes,
                              'gt_overlaps': overlaps,
                              'flipped': False,
                              'blurred': False,
                              'image_size': imsize,
                              'file_path': os.path.join(self._imgs_path, fp)})

        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """

        cache_file = os.path.join(self.cache_path,
                                  '%s_%s_roidb_ss.pkl' % (self.name, self._split))

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self.test_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def is_evaluated(self,img_ind, output_dir='./output/'):
        img_path = self._image_paths[img_ind]
        img_name = os.path.basename(img_path)
        img_dir = img_path[:img_path.find(img_name) - 1]
        res_dir = os.path.join(output_dir, img_dir)
        txt_fname = os.path.join(output_dir, img_dir, img_name.replace('jpg', 'txt'))

        if os.path.exists(txt_fname):
            print('[*] exists: {}'.format(txt_fname))
            return True

        return False

    def get_bbox(self,img_ind, output_dir='./output/'):
        img_path = self._image_paths[img_ind]
        img_name = os.path.basename(img_path)
        img_dir = img_path[:img_path.find(img_name) - 1]
        res_dir = os.path.join(output_dir, img_dir)
        txt_fname = os.path.join(output_dir, img_dir, img_name.replace('jpg', 'txt'))

        if os.path.exists(txt_fname):
            with open(txt_fname) as f:
                lines = f.readlines()
                dets = []
                for l in lines[2:]:
                    det = [float(x) for x in l.strip().split(' ')]
                    det[2] += det[0]
                    det[3] += det[1]
                    dets.append(det)
                return np.array(dets,dtype=np.float32)
        else:
            raise Exception('detection file is not there')

    def evaluate_single_detection(self, img_ind,all_boxes, output_dir='./output/',rois=None):
        img_path = self._image_paths[img_ind]

        img_name = os.path.basename(img_path)
        img_dir = img_path[:img_path.find(img_name) - 1]

        txt_fname = os.path.join(output_dir, img_dir, img_name.replace('jpg', 'txt'))

        res_dir = os.path.join(output_dir, img_dir)
        if not os.path.isdir(res_dir):
            os.makedirs(res_dir)
        
        with open(txt_fname, 'w') as f:
            f.write(img_path + '\n')
            f.write(str(len(all_boxes)) + '\n')
            for det in all_boxes:
                f.write('%d %d %d %d %g \n' % (
                    int(det[0]), int(det[1]), int(det[2]) - int(det[0]), int(det[3]) - int(det[1]),
                    det[4]))

        print('[#] wrote to %d file: %s' % (img_ind, txt_fname))

    def evaluate_detections(self, all_boxes, output_dir='./output/'):

        ctr = 0
        for i in range(len(self._image_paths)):
            img_path = self._image_paths[i]

            img_name = os.path.basename(img_path)
            img_dir = img_path[:img_path.find(img_name) - 1]

            txt_fname = os.path.join(output_dir, img_dir, img_name.replace('jpg', 'txt'))

            res_dir = os.path.join(output_dir, img_dir)
            if not os.path.isdir(res_dir):
                os.makedirs(res_dir)
            print('writing to %d file: %s' % (i, txt_fname))
            with open(txt_fname, 'w') as f:
                f.write(img_path + '\n')
                f.write(str(len(all_boxes[1][i])) + '\n')
                for det in all_boxes[1][i]:
                    f.write('%d %d %d %d %g \n' % (
                        int(det[0]), int(det[1]), int(det[2]) - int(det[0]), int(det[3]) - int(det[1]),
                        det[4]))


if __name__ == '__main__':
    d = wider('train')
    res = d.roidb
    from IPython import embed;

    embed()
