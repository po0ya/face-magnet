display_help() {
    echo "Usage: $0 <model-name> <image-path>" >&2
    # echo some stuff here for the -a or --add-options
    exit 1
}

if ! [ -n "$1" ] ; then  display_help ; exit 1 ; fi


set -x
set -e

export PYTHONUNBUFFERED="True"
PYTHONPATH=`pwd`/caffe-fast-rcnn/python:`pwd`/lib:`pwd`/tools:$PYTHONPATH

GPU_ID=0
NET=VGG16
NET_lc=${NET,,}
FILEPATH=$2

array=( $@ )
len=${#array[@]}
start=2
EXTRA_ARGS=${array[@]:$start:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}
MODEL_NAME=$1
SOLVER=models/wider/${NET}/solvers/${MODEL_NAME}.prototxt
CFG=experiments/cfgs/min800.yml
PROTOPREFIX=`python ./experiments/scripts/aux_py_scripts/proto_prefix.py ${SOLVER}`
SAVEDIR=output/`python ./experiments/scripts/aux_py_scripts/save_dir.py ${CFG}`/wider_train_${PROTOPREFIX}
CFGPREFIX=`python ./experiments/scripts/aux_py_scripts/cfg_prefix.py ${CFG}`
MODEL_PATH=${SAVEDIR}/${PROTOPREFIX}_iter_38000.caffemodel
TESTPROTO=`python ./experiments/scripts/aux_py_scripts/get_test_proto.py ${SOLVER}`

if ! [[ -a $MODEL_PATH ]]
then
    echo "[!] Model not found"
    exit
fi

echo "[#] Loading $MODEL_PATH and testing"
NET_FINAL=${MODEL_PATH}

python ./tools/test_net.py --gpu $GPU_ID \
  --def ${TESTPROTO} \
  --net ${NET_FINAL} \
  --imdb ${FILEPATH} \
  --cfg $CFG \
  --single \
  ${EXTRA_ARGS}
