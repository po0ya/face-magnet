if ! [ -n "$1" ] ; then echo "USAGE: $0 {default|context|facemagnet(_nc)|skipface(_nc)|sizesplit(_nc)} <GPUS[0,1]>" ; exit 1 ; fi
set -x
GPUS=$2
MODEL=$1
POST=$3
LOG_PATH=debug/train_${MODEL}.txt
./experiments/scripts/train_base.sh $GPUS VGG16 $1 38000 experiments/cfgs/min800.yml data/imagenet_models/VGG16.v2.caffemodel &> $LOG_PATH   &
echo $LOG_PATH
echo "process: " $! 
