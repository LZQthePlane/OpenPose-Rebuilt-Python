import cv2 as cv
import os
import argparse
import sys
from imutils.video import FPS
from pathlib import Path


FilePath = Path.cwd()
outFilePath = Path(FilePath / "test_out/")
threshold = 0.2
input_width, input_height = 368, 368

proto_file = str(FilePath/'model/pose_deploy.prototxt')
weights_file = str(FilePath/"model/pose_iter_102000.caffemodel")
nPoints = 22
POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9],
              [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17],
              [17, 18], [18, 19], [19, 20]]

# Usage example:  python openpose.py --video=test.mp4
parser = argparse.ArgumentParser(description='OpenPose for hand skeleton in python')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()


def choose_run_mode():
    global outFilePath
    if args.image:
        # Open the image file
        if not os.path.isfile(args.image):
            print("Input image file ", args.image, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(args.image)
        outFilePath = str(outFilePath / (args.image[:-4] + '_out.jpg'))
    elif args.video:
        # Open the video file
        if not os.path.isfile(args.video):
            print("Input video file ", args.video, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(args.video)
        outFilePath = str(outFilePath/(args.video[:-4]+'_out.mp4'))
    else:
        # Webcam input
        cap = cv.VideoCapture(0)
        outFilePath = str(outFilePath / 'webcam_out.mp4')
    return cap


def load_pretrain_model():
    # 加载预训练后的POSE的caffe模型
    net = cv.dnn.readNetFromCaffe(proto_file, caffeModel=weights_file)
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
        cv.putText(frame, fps_label, (0, origin_h - 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


if __name__ == "__main__":
    cap = choose_run_mode()
    net = load_pretrain_model()
    fps = FPS().start()
    vid_writer = cv.VideoWriter(outFilePath, cv.VideoWriter_fourcc(*'mp4v'), 30,
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                                 round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print("Output file is stored as ", outFilePath)
            cv.waitKey(3000)
            break

        origin_h, origin_w = frame.shape[:2]
        blob = cv.dnn.blobFromImage(frame, 1.0 / 255, (input_width, input_height), 0, swapRB=False, crop=False)
        net.setInput(blob)
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
                cv.circle(frame, (x, y), 6, (255, 255, 0), -1, cv.FILLED)
                cv.putText(frame, "{}".format(i), (x, y-15), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv.LINE_AA)
                points.append((x, y))
            else:
                points.append(None)

        # 画骨架
        for pair in POSE_PAIRS:
            A, B = pair[0], pair[1]
            if points[A] and points[B]:
                cv.line(frame, points[A], points[B], (0, 255, 255), 3, cv.LINE_AA)
                # cv.circle(frame, points[A], 8, (0, 0, 255), thickness=-1, lineType=cv.FILLED)
                # cv.circle(frame, points[B], 8, (0, 0, 255), thickness=-1, lineType=cv.FILLED)

        # Write the frame with the detection boxes
        if args.image:
            cv.imwrite(outFilePath, frame)
        else:
            vid_writer.write(frame)

        winName = 'Hand Skeleton from OpenPose'
        # cv.namedWindow(winName, cv.WINDOW_NORMAL)
        cv.imshow(winName, frame)

    if not args.image:
        vid_writer.release()
        cap.release()
    cv.destroyAllWindows()