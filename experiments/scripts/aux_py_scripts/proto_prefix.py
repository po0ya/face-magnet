import sys

import _init_paths
from caffe.proto import caffe_pb2
import google.protobuf.text_format as text_format
import google.protobuf as pb2

from fast_rcnn.config import get_snapshot_prefix

if __name__ == '__main__':
    snap_prefix = get_snapshot_prefix(sys.argv[1])
    print(snap_prefix)


