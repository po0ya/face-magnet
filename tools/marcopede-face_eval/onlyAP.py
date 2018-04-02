import sys
from database import *
import VOCpr
import os
import numpy as np
from loadData import loadDetections
from getColorLabel import *
from VOCpr import evaluate_optim, filterdet
import argparse
def load_fddb(ff):
    #print ff
    ftxt = open(ff, "rb")
    if ff == "detections/fddb/OlaworksDiscROC.txt":
        data = numpy.genfromtxt(ftxt, delimiter="\t")
    else:
        data = numpy.genfromtxt(ftxt, delimiter=" ")
    data.sort(axis=0)
    data = data[::-1]
    ln = os.path.splitext(ff)
    ln = os.path.basename(ln[0])
    if ln[-1] == "_":
        ln = ln[:-1]
    label = ln
    #pp = label.find("_DiscROC")
    #label = label[:pp]
    #if label[-1] == "_":
    #    label = label[:-1]
    color, label = getColorLabel(label)
    pos = numpy.where(data[:, 1] < 1000)[0][0]
    pr = data[pos, 0]
    #print pr
    label = label + " (%0.3f)" % (pr)
    plotid, = pylab.plot(data[:, 1], data[:, 0], color=color, linewidth=5)
    return plotid, pr, label

if __name__ == '__main__':


    parser = argparse.ArgumentParser(
        description='Plots AP curves on AFW/PASCAL faces')
    parser.add_argument(
        '--detfile', type=str, default="", help='Detection file')
    parser.add_argument(
        '--dataset', default="PASCAL", help='Select the dataset (AWF,PASCAL)')
    parser.add_argument(
        '--oldAnn', help='Use old annotations', action="store_true")
    parser.add_argument(
        '--minw', type=int, default=30, help='Minimum wide of a detectiion')
    parser.add_argument(
        '--minh', type=int, default=30, help='Minimum height of a detectiion')
    parser.add_argument('--nit', type=int, default=5,
                        help='Number of iterations for the bounding box refinement')

    args = parser.parse_args()
    minw = args.minw
    minh = args.minh
    nit = args.nit

    detfile = args.detfile
    if detfile == "":
    	raise ValueError('Please determine the detection file!')

    minpix = int(np.sqrt(0.5 * minw * minh))
    # Determine whether we are dealing with fddb or not
    if detfile.lower().find('pascal')!=-1:
    	tsImages = getRecord(
            PASCALfaces(minw=minw, minh=minh, useOldAnn=args.oldAnn), 10000)
    	ovr = 0.5
        is_point = False
        dets = loadDetections(detfile)
        dets = filterdet(dets, minpix)
        color, label = getColorLabel(detfile)
        r = evaluate_optim(
            tsImages, dets, label, color, point=is_point, iter=nit, ovr=ovr)
        print('{}'.format(r[0]))

    elif detfile.lower().find('afw')!=-1:
    	tsImages = getRecord(
            AFW(minw=minw, minh=minh, useOldAnn=args.oldAnn), 10000)
    	ovr = 0.5
        is_point = False
        dets = loadDetections(detfile)
        dets = filterdet(dets, minpix)
        color, label = getColorLabel(detfile)
        r = evaluate_optim(
            tsImages, dets, label, color, point=is_point, iter=nit, ovr=ovr)
        print('{}'.format(r[0]))

    elif detfile.lower().find('fddb')!=-1:
    	pid, pr, l = load_fddb(detfile)
    	print('{}'.format(pr))
    else:
    	raise ValueError('Unknown Dataset')


