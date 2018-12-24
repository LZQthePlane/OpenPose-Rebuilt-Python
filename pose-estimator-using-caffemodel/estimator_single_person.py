import cv2 as cv
import os
import argparse
import sys
from imutils.video import FPS
from pathlib import Path

# 相关参数初始化
FilePath = Path.cwd()
out_file_path = Path(FilePath / "test_out/")
threshold = 0.1
input_width, input_height = 368, 368
lineWidthMultiper = 2

proto_file = ""
weights_file = ""
nPoints = 0
POSE_PAIRS = []


def choose_pose(model='COCO'):
    """
    选择输出的骨架模型, 可选：MPI / COCO / BODY_25
    """
    global proto_file, weights_file, nPoints, POSE_PAIRS
    if model is "COCO":
        proto_file = str(FilePath / "model/coco/pose_deploy_linevec_faster_4_stages.prototxt")
        weights_file = str(FilePath / "model/coco/pose_iter_440000.caffemodel")
        nPoints = 18
        POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
                      [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]
    elif model is "MPI":
        proto_file = str(FilePath / "model/mpi/pose_deploy_linevec_faster_4_stages.prototxt")
        weights_file = str(FilePath / "model/mpi/pose_iter_440000.caffemodel")
        nPoints = 15
        POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11],
                      [11, 12], [12, 13]]
    elif model is "BODY_25":
        proto_file = str(FilePath / "model/body_25/pose_deploy.prototxt")
        weights_file = str(FilePath / "model/body_25/pose_iter_584000.caffemodel")
        nPoints = 25
        POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [10, 11], [11, 24],
                      [11, 22], [22, 23], [8, 12], [12, 13], [13, 14], [14, 21], [14, 19], [19, 20], [0, 15], [15, 17],
                      [0, 16], [16, 18]]


def choose_run_mode():
    """
    选择输入：image/video/webcam
    """
    global out_file_path
    if args.image:
        # Open the image file
        if not os.path.isfile(args.image):
            print("Input image file ", args.image, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(args.image)
        out_file_path = str(out_file_path / (args.image[:-4] + '_out.jpg'))
    elif args.video:
        # Open the video file
        if not os.path.isfile(args.video):
            print("Input video file ", args.video, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(args.video)
        out_file_path = str(out_file_path/(args.video[:-4]+'_out.mp4'))
    else:
        # Webcam input
        cap = cv.VideoCapture(0)
        out_file_path = str(out_file_path / 'webcam_out.mp4')
    return cap


def load_pretrain_model():
    """
    加载预训练后的POSE的caffe模型
    """
    net = cv.dnn.readNetFromCaffe(proto_file, weights_file)
    print('POSE caffe model loaded successfully')
    # 调用GPU模块，但目前opencv仅支持intel GPU
    # tested with Intel GPUs only, or it will automatically switch to CPU
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)
    return net


def show_fps():
    if not args.image:
        fps.update()
        fps.stop()
        fps_label = "FPS: {:.2f}".format(fps.fps())
        cv.putText(frame, fps_label, (0, origin_h - 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), lineWidthMultiper)


if __name__ == "__main__":
    # Usage example:  python estimator_single_person.py --video=test.mp4
    parser = argparse.ArgumentParser(description='OpenPose for pose skeleton in python')
    parser.add_argument('--image', help='Path to image file.')
    parser.add_argument('--video', help='Path to video file.')
    args = parser.parse_args()

    choose_pose('COCO')
    cap = choose_run_mode()
    net = load_pretrain_model()
    fps = FPS().start()
    vid_writer = cv.VideoWriter(out_file_path, cv.VideoWriter_fourcc(*'mp4v'), 30,
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                                 round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print("Output file is stored as ", out_file_path)
            cv.waitKey(3000)
            break

        origin_h, origin_w = frame.shape[:2]
        # 首先，我们将像素值标准化为（0,1）。然后我们指定图像的尺寸。接下来，要减去的平均值，即（0,0,0）
        # 不同算法及训练模型的blobFromImage参数不同，可访问opencv的github地址查询
        # https://github.com/opencv/opencv/tree/master/samples/dnn
        blob = cv.dnn.blobFromImage(frame, 1.0 / 255, (input_width, input_height), 0, swapRB=False, crop=False)
        net.setInput(blob)
        # 第一个维度是图像ID（如果将多个图像传递给网络）。
        # 第二个维度表示关节点的索引。该模型生成Confidence Maps和Part Affinity Maps。
        # 对于COCO模型，它由57个部分组成: 18个关键点置信度map + 1个背景 + 19 * 2部分亲和力图。
        # 第三个维度是输出映射map的高度。
        # 第四个维度是输出映射map的宽度。
        detections = net.forward()
        H = detections.shape[2]
        W = detections.shape[3]
        # 存储关节点
        points = []

        for i in range(nPoints):
            probility_map = detections[0, i, :, :]
            #
            min_value, confidence, min_loc, point = cv.minMaxLoc(probility_map)
            #
            x = int(origin_w * (point[0] / W))
            y = int(origin_h * (point[1] / H))

            if confidence > threshold:
                cv.circle(frame, (x, y), lineWidthMultiper*3, (0, 255, 255), -1, cv.FILLED)
                # cv.putText(frame, "{}".format(i), (x, y-15), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv.LINE_AA)
                points.append((x, y))
            else:
                points.append(None)

        # 画骨架
        for pair in POSE_PAIRS:
            A, B = pair[0], pair[1]
            if points[A] and points[B]:
                cv.line(frame, points[A], points[B], (0, 0, 255), lineWidthMultiper, cv.LINE_AA)
                # cv.circle(frame, points[A], 8, (0, 0, 255), thickness=-1, lineType=cv.FILLED)
                # cv.circle(frame, points[B], 8, (0, 0, 255), thickness=-1, lineType=cv.FILLED)
        # 显示实时FPS
        show_fps()

        # Write the frame with the detection boxes
        if args.image:
            cv.imwrite(out_file_path, frame)
        else:
            vid_writer.write(frame)

        winName = 'Pose Skeleton from OpenPose'
        # cv.namedWindow(winName, cv.WINDOW_NORMAL)
        cv.imshow(winName, frame)

    if not args.image:
        vid_writer.release()
        cap.release()
    cv.destroyAllWindows()
