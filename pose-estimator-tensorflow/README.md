
## Usage Examples :
Creat a file names `graph_model_coco`;   
Put the test file (iamge or video) under the same directory of `openpose_tf.py`.   
   
 - `python3 handpose.py --image=test.jpg`   
 - `python3 handpose.py --video=test.mp4`   
 - if no argument provided, it starts the webcam.
 
## Pretrained models intro
 - **graph_opt.pb**: training with the VGG net, as same as the CMU providing caffemodel.
 - **graph_opt_mobile.pb**: 

## Acknowledgement
Thanks [ildoonet](https://github.com/ildoonet) and his awesome work [tf-pose-estimation](), the graph weight files are collected there.
 
