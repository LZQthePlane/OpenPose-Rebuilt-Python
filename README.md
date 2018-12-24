# OpenPose Rebuilt-Python   
Rebuilting the CMU-OpenPose pose estimatior using Python with OpenCV and Tensorflow.  
(The code comments are partly descibed in chinese)   

## Pretrained-model Downloading   
In this work, I used both caffemodel and tensorflow-graph-model, you can download them from [here](https://pan.baidu.com/s/1XT8pHtNP1FQs3BPHgD5f-A)
## Requirements : 
1. OpenCV > 3.4.1
2. TensorFlow > 1.2.0
3. imutils

Difference between BODY_25 vs. COCO vs. MPI
COCO model will eventually be removed. BODY_25 model is faster, more accurate, and it includes foot keypoints. However, COCO requires less memory on GPU (being able to fit into 2GB GPUs with the default settings) and it runs faster on CPU-only mode. MPI model is only meant for people requiring the MPI-keypoint structure. It is also slower than BODY_25 and far less accurate.

Output Format
