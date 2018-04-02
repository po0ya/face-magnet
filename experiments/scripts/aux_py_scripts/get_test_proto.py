import _init_paths
import sys
import caffe
from caffe.proto import caffe_pb2
import google.protobuf.text_format as text_format

import google.protobuf as pb2

if __name__ == '__main__':

    solver_param = caffe_pb2.SolverParameter()

    with open(sys.argv[1], 'rt') as f:
        text_format.Merge(f.read(), solver_param)

    print(solver_param.train_net.replace('train','test'))
