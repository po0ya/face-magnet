display_help() {
    echo "Usage: $0 {net[VGG16]} {dataset[wider|wider_test]} {solver} {num iterations} {config}" >&2
    # echo some stuff here for the -a or --add-options 
    exit 1
}

if ! [ -n "$1" ] ; then  display_help ; exit 1 ; fi


set -x
set -e

export PYTHONUNBUFFERED="True"
PYTHONPATH=`pwd`/caffe-fast-rcnn/python:`pwd`/lib:`pwd`/tools:$PYTHONPATH

GPU_ID=0
NET=$1
NET_lc=${NET,,}
DATASET=$2

array=( $@ )
len=${#array[@]}
start=5
EXTRA_ARGS=${array[@]:$start:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}
POST=$3
case $DATASET in
   pascal)
    TRAIN_IMDB="wider_train"
    TEST_IMDB="pacal"
    PT_DIR="wider"
    ITERS=$4
    ;;

   fddb)

    TRAIN_IMDB="wider_train"
    TEST_IMDB="fddb"
    PT_DIR="wider"
    ITERS=$4
    ;;
  wider)
    TRAIN_IMDB="wider_train"
    TEST_IMDB="wider_val"
    PT_DIR="wider"
    ITERS=$4
    ;;
  wider_test)
    TRAIN_IMDB="wider_train"
    TEST_IMDB="wider_test"
    PT_DIR="wider"
    ITERS=$4
    ;;
*)
    echo "[!] No dataset given"
    exit
    ;;
esac
SOLVER=models/${PT_DIR}/${NET}/solvers/${POST}.prototxt
CFG=$5
PROTOPREFIX=`python ./experiments/scripts/aux_py_scripts/proto_prefix.py ${SOLVER}`
SAVEDIR=output/`python ./experiments/scripts/aux_py_scripts/save_dir.py ${CFG}`/${TRAIN_IMDB}
CFGPREFIX=`python ./experiments/scripts/aux_py_scripts/cfg_prefix.py ${CFG}`
MODEL_PATH=${SAVEDIR}/${PROTOPREFIX}_iter_${ITERS}.caffemodel
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
  --imdb ${TEST_IMDB} \
  --cfg $CFG \
  ${EXTRA_ARGS}
