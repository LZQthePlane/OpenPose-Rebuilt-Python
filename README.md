# OpenPose Rebuilt-Python   
Rebuilting the CMU-OpenPose pose estimatior using Python with OpenCV and Tensorflow.  
(The code comments are partly descibed in chinese)   

-------
## Pretrained-model Downloading   
In this work, I used both **caffemodel** and **tensorflow-graph-model**, you can download them from [here](https://pan.baidu.com/s/1XT8pHtNP1FQs3BPHgD5f-A), Then place the pretrained models to corresponding directory respectively.   
### *Examples:*  
 - place `caffe_models\pose\body_25\pose_iter_584000.caffemodel` into `pose-estimator-using-caffemodel\model\body_25\`    
 - place `caffe_models\hand\pose_iter_102000.caffemodel` into `hand-estimator-using-caffemodel\model\`   
 - place `openpose graph model coco\graph_opt.pb` into `pose-estimator-tensorflow\graph_model_coco\`   

-------
## Requirements : 
 - OpenCV > 3.4.1
 - TensorFlow > 1.2.0
 - imutils

-------
## Usage:
See the sub-README.md in sub-folder.   

-------
## BODY_25 vs. COCO vs. MPI
 - BODY_25 model is ***faster, more accurate***, and it includes foot keypoints. 
 - COCO requires less memory on GPU (being able to fit into 2GB GPUs with the default settings) and it runs ***faster on CPU-only mode***. 
 - MPI model is only meant for people requiring the MPI-keypoint structure. It is also slower than BODY_25 and far less accurate.
### *Output Format*   
**Body_25 in left, COCO in middle, MPI in right.**
<div style="float:left;border:solid 5px 000;margin:2px;"><img src="https://github.com/LZQthePlane/OpenPose-Rebuilt-Python/blob/master/test_out/Output-BODY_25.jpg" width="290"/><img src="https://github.com/LZQthePlane/OpenPose-Rebuilt-Python/blob/master/test_out/Output-COCO.jpg"  width="290"/><img src="https://github.com/LZQthePlane/OpenPose-Rebuilt-Python/blob/master/test_out/Output-MPI.jpg" width="290"></div>
**more details** [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md), **Hand Output Format** included as well.

-------
## Results Showing
