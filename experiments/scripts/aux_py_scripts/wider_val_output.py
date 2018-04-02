import yaml
import os
import sys
import argparse

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')

    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    net_name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    print net_name
