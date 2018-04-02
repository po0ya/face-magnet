#!/usr/bin/env bash
set -x
set -e
MODEL=facemagnet
ITER=38000
POST=wider_test
CFG=experiments/cfgs/min800.yml
NUM_GPUS=$(( `nvidia-smi --query-gpu=gpu_name,gpu_bus_id,vbios_version --format=csv | wc -l` - 1 ))
SCALES=(1 1.5 2)
LOG_DIR="debug/min800/${MODEL}_${POST}"
mkdir -p $LOG_DIR
echo "[*] Successfully made the log dir"

PYR_DET=""
for scale in "${SCALES[@]}" ; do 
    cur_pids=""
    for i in `seq 0 $(( NUM_GPUS - 1 ))`; do
        export CUDA_VISIBLE_DEVICES=$i
        LOG_FILE="${LOG_DIR}/scale_iter${ITER}_${scale}_${i}.log"
        ./experiments/scripts/test_base.sh VGG16 wider_test $MODEL $ITER $CFG --orig_scale $scale --max_size 3000 --shuffle &> $LOG_FILE &
        cur_pid=$!
        echo "process: " $!
        cur_pids="$cur_pids $cur_pid"
        sleep 3
    done
    for pid in $cur_pids;
    do 
        wait $pid
    done
    echo "[*] Detections are done! Now MATLAB EVAL!"
    sleep 10
    LOG_FILE="${LOG_DIR}/matlab_scale_ITER${ITER}_${scale}.log"
    ./experiments/scripts/test_base.sh VGG16 wider_test $MODEL $ITER $CFG --orig_scale $scale --max_size 3000 --matlab_eval &> $LOG_FILE &
    echo "process: " $! "Log file: " $LOG_FILE
    wait
    DET_FILE="$(./experiments/scripts/test_base.sh VGG16 wider $MODEL $ITER $CFG --orig_scale $scale --max_size 3000 --output_dir | tail -n 1)/detections.pkl"
    PYR_DETS="$DET_FILE $PYR_DETS"
done
PYR_DET_FILE="$(./experiments/scripts/test_base.sh VGG16 wider $MODEL $ITER $CFG --max_size 3000 --output_dir | tail -n 1)_pyramid_10_15_20"
python tools/merge_dets.py $PYR_DET_FILE $PYR_DETS
