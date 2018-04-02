#!/usr/bin/env bash
display_help() {
    echo "Usage: $0 {DETECTION_OUTPUT [run test_base.sh with fddb]} {LABEL}" >&2
    # echo some stuff here for the -a or --add-options 
    exit 1
}

if ! [ -n "$1" ] ; then  display_help ; exit 1 ; fi


set -x
set -e
OUTPUT=$1
NAME=$2
CUR_DIR=`pwd`
FDDB_DIR=
#NAME=`expr match "$OUTPUT" '.*_wider_\(.*\).*.txt'`
cp $OUTPUT ${FDDB_DIR}/evaluation/ttt.txt
cd ${FDDB_DIR}/evaluation/

pwd
./evaluate -a allEllipseList.txt \
    -d ttt.txt \
    -l ../img_list.csv \
    -i ../originalPics/

cp tempDiscROC.txt ${CUR_DIR}/tools/marcopede-face_eval/detections/fddb_disc/${NAME}.txt
cp tempContROC.txt ${CUR_DIR}/tools/marcopede-face_eval/detections/fddb_cont/${NAME}.txt

cd ${CUR_DIR}/tools/marcopede-face_eval/
python plot_AP_fddb.py
python plot_AP_fddb.py --dataset fddb_disc
