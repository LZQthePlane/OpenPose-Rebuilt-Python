
A.Requirements : 
1. OpenCV > 3.4.1
2. Matplotlib for Notebook
3. RUN getModels.sh from command line Or Download caffe model from http://posefs1.perception.cs.cmu.edu/Users/ZheCao/pose_iter_440000.caffemodel and put it in pose/coco folder


Difference between BODY_25 vs. COCO vs. MPI
COCO model will eventually be removed. BODY_25 model is faster, more accurate, and it includes foot keypoints. However, COCO requires less memory on GPU (being able to fit into 2GB GPUs with the default settings) and it runs faster on CPU-only mode. MPI model is only meant for people requiring the MPI-keypoint structure. It is also slower than BODY_25 and far less accurate.

Output Format
