# *Face-MagNet*: Magnifying Feature Maps to Detect Small Faces
 
By Pouya Samangouei\*, Mahyar Najibi\*, Larry Davis, Rama Chellappa

\* contributed equally.

This Python implementation is based on [Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn).

#### License

Face-MagNet is released under the Apache 2 License (refer to the LICENSE file for details).

#### Citing Face-MagNet

If you find Faster R-CNN useful in your research, please consider citing:

    @inproceedings{wacv18facemagnet,
        Author = {Pouya Samangouei and Mahyar Najibi and Larry Davis and Rama Chellappa},
        Title = {Face-MagNet: Magnifying Feature Maps to Detect Small Faces},
        Booktitle = {IEEE Winter Conf. on Applications of Computer Vision ({WACV})},
        Year = {2018}
    }

    
### Installation

- Download `face-magnet`. 
```
cd <project-parent-dir>
git clone --recursive https://github.com/po0ya/face-magnet
FM_ROOT=`pwd`/face-magnet
```

- Install python requirements.
    ```
    cd $FM_ROOT
    pip install -r requirements.txt
    ```
- Make `lib`:
    ```
    cd $FM_ROOT/lib
    make
    ```
    
- Install `nccl` from `https://developer.nvidia.com/nccl` for multi-gpu training.

- Build `caffe`.
    ```
    cd caffe-facemagnet
    # Set build flags e.g., edit caffe-facemagnet/Makefile.config appropriately.
    make -j8
    make pycaffe
    ```

- Download the [`WIDER-Face`](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) dataset. 
    ```
    WIDER_PATH=<path-to-wider-face>
    ```

- Perpare the `data` directory:
    - Download `imagenet` pretrained weights using scripts from [Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn):
        ```
        ./data/scripts/fetch_imagenet_models.sh
        ```
    
    - Create a symlink to `WIDER-Face` path:
        ```
        ln -s $WIDER_PATH data/wider
        ```
        - (optional) Copy `imglist.csv` files into `WIDER_PATH`:
        ```
        cp -r data/WIDER_imglists/* $WIDER_PATH
        ```
        These `csv` files are generated with `matlab/wider_csv.m` and is provided
        in case `matlab` is not available. If it is, the code will generate these
        automatically.
         
- Create a or link a `debug` directory for output logs.
    ```
    mkdir debug
    ```
    or
    ```
    ln -s <path-to-debug-dir> debug
    ```
    
### Train a model:
```
./experiments/scripts/train_min800.sh <model-name> <gpu-ids>
```
- `<model-name>` can be:
    - `facemagnet`: Face-MagNet model for the 8th row of Table 1 and the final results.
    - `sizesplit`: SizeSplit model for the 7th row of Table 1.
    - `skipface`: SkipFace model for the 6th row of Table 1.
    - `context`: Context model for the 5th row of Table 1.
    - `default`: For the first row of Table 1.
    - `facemagnet_nc`, `sizesplit_nc`, and `skipface_nc` for the models without context pooling.
    
- `<gpu-ids>` is a string of comma separated GPU ids.
- To see the training progress `tail -f <log-path>` where `<log-path>` is shown
after running the above script.

For example, for `facemagnet` run:
```
./experiments/scripts/train_min800.sh facemagnet 0,1
```
To train Face-MagNet on two GPUs.

For single GPU training please use `./tools/train_net.py`.
```
 python ./tools/train_net.py --gpu <gpu-id> \
  --solver models/wider/VGG16/solvers/<model-name>.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb wider_train \
  --iters 76000 \
  --cfg experiments/cfgs/min800.yml \
  ${EXTRA_ARGS}
```
### Single image detection

To detect faces on a single image with a trained model:
```
./experiments/scripts/test_single.sh <model-name> <image-path> <args>
```

- `<model-name>` is the same as the training section.
- `<image-path>` is the path to the image
- `args` can be the following optional arguments:
    - `--min_size <int>` Overrides `cfg.TEST.SCALES` so that the minimum dimension of the image is scaled to this value.
    - `--max_size <int>` Overrides `cfg.TEST.MAX_SIZE` so that the maximum dimension is not
    greater than its value. 
    - `--pyramid` Performs the detection three times on scales `[0.5, 1, 2]` of the image and combines the result.

For example to detect the faces in images of `data/demo`:

```
./experiments/scripts/test_single.sh facemagnet data/demo/demo0.jpg --pyramid
```

### Reproducing the results
- To produce the results of each row of Table 1, use:
    ```
    ./experiments/scripts/test_min800.sh <model-name>
    ```

- To perform the benchmark test on WIDER-Face:
    ```
    ./experiments/scripts/test_wider.sh
    ```


- To test the `fddb` and `pascal` datasets:
    - Pascal image list can be found in `data/pascal_imglist.csv`
    - Download [FDDB](http://vis-www.cs.umass.edu/fddb/) and link it to `data/fddb`.
    - Prepare Pascal Faces and put it under `data/pascal`.
    - To get the detections:
        ```
        ./experiments/scripts/test_base.sh VGG16 {fddb|pascal} <model-name> 38000 ./experiments/cfgs/min800.yml
        ```
    - To produce eval plots see `tools/marcopede-face_eval`.
    - Take a look at and use `./experiments/scripts/plot_fddb.sh` for FDDB. 

### Notes
- The results are reported using an NVIDIA P6000 GPU with 24GB GPU memory. Big images need 
a higher memory because of the sizes of the convolutional feature maps, therefore if you're facing memory
issues try setting `cfg.TEST.MAX_SIZE` to a smaller number. 

##### TODOs: 
- Uploading the models.
