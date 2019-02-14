
## Usage Examples :
***for `openpose_tf.py`***
1. Creat a file names `graph_model_coco`, and palce the pretrained model here;   
2. Put the test file (image or video) under the same directory of `openpose_tf.py`.   
3. Commmand line input:   
 - `python openpose_tf.py --image=test.jpg`   
 - `python openpose_tf.py --video=test.mp4`   
 - if no argument provided, it starts the webcam.

***for `openpose_skeleton_sequence_drawer.py`***
1. Same as the upper;   
2. Commmand line input:   
 - `python openpose_skeleton_sequence_drawer.py`

## Pretrained models intro
 - **graph_opt.pb**: training with the VGG net, as same as the CMU providing caffemodel, ***more accurate but slower***
 - **graph_opt_mobile.pb**:  training with the Mobilenet, much smaller than the origin VGG, ***faster but less accurate***

## Acknowledgement
Thanks [ildoonet](https://github.com/ildoonet) and his awesome work [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation), the graph weight files are collected there.
