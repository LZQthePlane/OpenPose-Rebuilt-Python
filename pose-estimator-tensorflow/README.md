
## Usage Examples :
Creat a file names `graph_model_coco`, and palce the pretrained model here;   
Put the test file (iamge or video) under the same directory of `openpose_tf.py`.   
   
 - `python3 openpose_tf.py --image=test.jpg`   
 - `python3 openpose_tf.py --video=test.mp4`   
 - if no argument provided, it starts the webcam.
 
## Pretrained models intro
 - **graph_opt.pb**: training with the VGG net, as same as the CMU providing caffemodel, ***more accurate but slower***
 - **graph_opt_mobile.pb**:  training with the Mobilenet, much smaller than the origin VGG, ***faster but less accurate***

## Acknowledgement
Thanks [ildoonet](https://github.com/ildoonet) and his awesome work [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation), the graph weight files are collected there.
