set -x
./experiments/scripts/release_scripts/train_base.sh 0,1 VGG16 facemagnet 38000 experiments/cfgs/min800.yml data/imagenet_models/VGG16.v2.caffemodel 
