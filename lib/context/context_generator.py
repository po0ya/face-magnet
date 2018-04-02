import caffe
import numpy as np
import yaml


class ContextGenerator(caffe.Layer):
    def setup(self, bottom, top):

        top[0].reshape(bottom[1].shape[0],5)
        try:
            layer_params = yaml.load(self.param_str_)
        except AttributeError:
            # Caffe compatibility issue
            layer_params = yaml.load(self.param_str)
        self._w_scale = layer_params['w_scale']
        self._h_scale = layer_params['h_scale']
        self._x_rel_pos = layer_params['x_rel_pos']
        self._y_rel_pos = layer_params['y_rel_pos']
        try:
            self._gt_roi_flag = layer_params['gt_roi']
        except KeyError:
            self._gt_roi_flag = False
        assert (0 <= self._x_rel_pos <= 1)
        assert (0 <= self._y_rel_pos <= 1)
        assert (0 <= self._h_scale )
        assert (0 <= self._w_scale )
    def reshape(self, bottom, top):
        # nothing to be done!
        pass

    def backward(self, top, propagate_down, bottom):
        # no need to backprop!
        pass

    def forward(self, bottom, top):
        # Get the maps as input
        ims = bottom[0].data
        rois = bottom[1].data
        if self._gt_roi_flag:
            context_rois = self._get_context_rois((ims.shape[3], ims.shape[2]), rois[:,:4])
            levels = np.zeros((rois.shape[0], 1), dtype=np.int)
        else:
            levels = bottom[1].data[:, 0]
            levels = levels[:, np.newaxis]
            context_rois = self._get_context_rois((ims.shape[3], ims.shape[2]), rois[:,1:])

        blob_out = np.hstack((levels, context_rois))

        top[0].reshape(*(blob_out.shape))
        top[0].data[...] = blob_out


    def _get_context_rois(self,im_size,im_rois):

        context_rois = np.zeros_like(im_rois)
        center_x = (im_rois[:, 0] + im_rois[:, 2]) / 2
        center_y = (im_rois[:, 1] + im_rois[:, 3]) / 2
        half_width = (center_x - im_rois[:, 0]) * self._w_scale
        half_height = (center_y - im_rois[:, 1]) * self._h_scale
        context_rois[:, 0] = np.max(np.hstack(
            (np.zeros([center_x.shape[0], 1]), (center_x - half_width*self._x_rel_pos*2)[:, np.newaxis])
        ), axis=1)
        context_rois[:, 1] = np.max(np.hstack(
            (np.zeros([center_x.shape[0], 1]), (center_y - half_height*self._y_rel_pos*2)[:, np.newaxis])
        ), axis=1)
        context_rois[:, 2] = np.min(np.hstack(
            (im_size[0] * np.ones([center_x.shape[0], 1]), (context_rois[:, 0] + half_width*2)[:, np.newaxis])
        ), axis=1)
        context_rois[:, 3] = np.min(np.hstack(
            (im_size[1] * np.ones([center_x.shape[0], 1]), (context_rois[:, 1] + half_height*2)[:, np.newaxis])
        ), axis=1)
        return context_rois



