
display_help() {
    echo "Usage: $0 {gpus} {net[VGG16]} {solver} {num iterations} {config}" >&2
    # echo some stuff here for the -a or --add-options 
    exit 1
}

if ! [ -n "$1" ] ; then  display_help ; exit 1 ; fi

set -x
set -e

export PYTHONUNBUFFERED="True"
PYTHONPATH=`pwd`/caffe-fast-rcnn/python:`pwd`/lib:`pwd`/tools:$PYTHONPATH

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=wider

POST=$3

TRAIN_IMDB="wider_train"
TEST_IMDB="wider_val"
PT_DIR="wider"
ITERS=$4

SOLVER=models/${PT_DIR}/${NET}/solvers/${POST}.prototxt
CFG=$5
array=( $@ )
len=${#array[@]}
start=7
EXTRA_ARGS=${array[@]:$start:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

PROTOPREFIX=`python ./experiments/scripts/aux_py_scripts/proto_prefix.py ${SOLVER}`
SAVEDIR=output/`python ./experiments/scripts/aux_py_scripts/save_dir.py ${CFG}`/${TRAIN_IMDB}
CFGPREFIX=`python ./experiments/scripts/aux_py_scripts/cfg_prefix.py ${CFG}`
MODEL_PATH=${SAVEDIR}/${PROTOPREFIX}_iter_${ITERS}.caffemodel
TESTPROTO=`python ./experiments/scripts/aux_py_scripts/get_test_proto.py ${SOLVER}`


LOG="experiments/logs/magnet_${NET}_${EXTRA_ARGS_SLUG}_${PROTOPREFIX}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

if ! [[ -a $MODEL_PATH ]]
then
 python ./tools/train_net_multigpu.py --gpus ${GPU_ID} \
  --solver ${SOLVER} \
  --weights $6 \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg $CFG \
  ${EXTRA_ARGS}
fi
