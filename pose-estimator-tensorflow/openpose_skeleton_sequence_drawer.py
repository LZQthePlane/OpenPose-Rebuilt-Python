import cv2 as cv
import numpy as np
import os
import math
import argparse
import sys
import itertools
import tensorflow as tf
from enum import Enum
from collections import namedtuple
from scipy.ndimage import maximum_filter, gaussian_filter
from imutils.video import FPS
from pathlib import Path
import time


class Human:
    """
    body_parts: list of BodyPart
    """
    __slots__ = ('body_parts', 'pairs', 'uidx_list')

    def __init__(self, pairs):
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        for pair in pairs:
            self.add_pair(pair)

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def add_pair(self, pair):
        self.pairs.append(pair)
        self.body_parts[pair.part_idx1] = BodyPart(Human._get_uidx(pair.part_idx1, pair.idx1),
                                                   pair.part_idx1,
                                                   pair.coord1[0], pair.coord1[1], pair.score)
        self.body_parts[pair.part_idx2] = BodyPart(Human._get_uidx(pair.part_idx2, pair.idx2),
                                                   pair.part_idx2,
                                                   pair.coord2[0], pair.coord2[1], pair.score)
        self.uidx_list.add(Human._get_uidx(pair.part_idx1, pair.idx1))
        self.uidx_list.add(Human._get_uidx(pair.part_idx2, pair.idx2))

    def is_connected(self, other):
        return len(self.uidx_list & other.uidx_list) > 0

    def merge(self, other):
        for pair in other.pairs:
            self.add_pair(pair)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])


class BodyPart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score


class TfPoseEstimator:
    ENSEMBLE = 'addup'  # average, addup

    def __init__(self, graph_path, target_size=(368, 368)):
        self.target_size = target_size

        # load graph
        with tf.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        self.graph = tf.get_default_graph()
        tf.import_graph_def(graph_def, name='TfPoseEstimator')
        self.persistent_sess = tf.Session(graph=self.graph)

        self.tensor_image = self.graph.get_tensor_by_name('TfPoseEstimator/image:0')
        self.tensor_output = self.graph.get_tensor_by_name('TfPoseEstimator/Openpose/concat_stage7:0')
        self.heatMat = self.pafMat = None

    @staticmethod
    def draw_pose_bbox(npimg, humans, imgcopy=False):
        if imgcopy:
            npimg = np.copy(npimg)
        image_h, image_w = npimg.shape[:2]
        joints, bboxes, xcenter = [], [], []
        for human in humans:
            xs, ys, centers = [], [], {}
            # 将所有关节点绘制到图像上
            for i in range(CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[i]
                center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))

                centers[i] = center
                xs.append(center[0])
                ys.append(center[1])
                # 绘制关节点
                cv.circle(npimg, center, 3, CocoColors[i], thickness=3, lineType=8, shift=0)
            # 将属于同一人的关节点按照各个部位相连
            for pair_order, pair in enumerate(CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue
                cv.line(npimg, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3, cv.LINE_AA)
            # 根据每个人的关节点信息生成ROI区域
            xmin = float(min(xs) / image_w)
            ymin = float(min(ys) / image_h)
            xmax = float(max(xs) / image_w)
            ymax = float(max(ys) / image_h)
            bboxes.append([xmin, ymin, xmax, ymax, 0.9999])
            joints.append(centers)
            if 1 in centers:
                xcenter.append(centers[1][0])

            # draw bounding_boxes
            # x_start, x_end= int(xmin*image_w) - 20, int(xmax*image_w) + 20
            # y_start, y_end =int(ymin*image_h) - 15, int(ymax*image_h) + 15
            # cv.rectangle(npimg, (x_start, y_start), (x_end, y_end), [0, 250, 0], 3)
        return npimg, joints, bboxes, xcenter

    @staticmethod
    def draw_pose_sequence(frame, humans, cnt):
        # global back_ground
        image_h, image_w = frame.shape[:2]
        for human in humans:
            xs, ys, centers = [], [], {}
            # 将所有关节点绘制到图像上
            joints_num = CocoPart.Background.value
            for i in range(joints_num):
                if i not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[i]
                center = (int(body_part.x * image_w + 0.5),
                          int(body_part.y * image_h + 0.5))

                pos_move = cnt * back_ground_w / 500
                center = (int(center[0] + pos_move / 2),
                          int(center[1] + back_ground_h / 1.5 - pos_move))

                centers[i] = center
                xs.append(center[0])
                ys.append(center[1])
                # 绘制关节点
                cv.circle(back_ground, centers[i], 3, CocoColors[i], thickness=3, lineType=8, shift=0)

            # 将属于同一人的关节点按照各个部位相连
            for pair_order, pair in enumerate(CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue
                cv.line(back_ground, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3, cv.LINE_AA)

    @staticmethod
    def draw_pose_sequence_reverse(frame, humans, cnt):
        image_h, image_w = frame.shape[:2]
        for human in humans:
            xs, ys, centers = [], [], {}
            # 将所有关节点绘制到图像上
            joints_num = CocoPart.Background.value
            for i in range(joints_num):
                if i not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[i]
                center = (int(body_part.x * image_w + 0.5),
                          int(body_part.y * image_h + 0.5))

                pos_move = cnt * back_ground_w / 500
                center = (int(center[0] + back_ground_w/1.5 - pos_move/2),
                          int(center[1] + pos_move))

                centers[i] = center
                xs.append(center[0])
                ys.append(center[1])
                # 绘制关节点
                cv.circle(back_ground, centers[i], 3, CocoColors[i], thickness=3, lineType=8, shift=0)

            # 将属于同一人的关节点按照各个部位相连
            for pair_order, pair in enumerate(CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue
                cv.line(back_ground, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3, cv.LINE_AA)

    @staticmethod
    def draw_pose_sequence_horizontally(frame, humans, cnt):
        image_h, image_w = frame.shape[:2]
        for human in humans:
            xs, ys, centers = [], [], {}
            # 将所有关节点绘制到图像上
            joints_num = CocoPart.Background.value
            for i in range(joints_num):
                if i not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[i]
                center = (int(body_part.x * image_w + 0.5),
                          int(body_part.y * image_h + 0.5))

                pos_move = cnt * back_ground_w / 200
                center = (int(center[0] + pos_move), center[1])

                centers[i] = center
                xs.append(center[0])
                ys.append(center[1])
                # 绘制关节点
                cv.circle(back_ground, centers[i], 3, CocoColors[i], thickness=3, lineType=8, shift=0)

            # 将属于同一人的关节点按照各个部位相连
            for pair_order, pair in enumerate(CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue
                cv.line(back_ground, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3, cv.LINE_AA)

    def inference(self, npimg):
        if npimg is None:
            raise Exception('The image does not exist.')

        rois = []
        infos = []
        # _get_scaled_img
        if npimg.shape[:2] != (self.target_size[1], self.target_size[0]):
            # resize
            npimg = cv.resize(npimg, self.target_size)
            rois.extend([npimg])
            infos.extend([(0.0, 0.0, 1.0, 1.0)])

        output = self.persistent_sess.run(self.tensor_output, feed_dict={self.tensor_image: rois})

        heat_mats = output[:, :, :, :19]
        paf_mats = output[:, :, :, 19:]

        output_h, output_w = output.shape[1:3]
        max_ratio_w = max_ratio_h = 10000.0
        for info in infos:
            max_ratio_w = min(max_ratio_w, info[2])
            max_ratio_h = min(max_ratio_h, info[3])
        mat_w, mat_h = int(output_w / max_ratio_w), int(output_h / max_ratio_h)

        resized_heat_mat = np.zeros((mat_h, mat_w, 19), dtype=np.float32)
        resized_paf_mat = np.zeros((mat_h, mat_w, 38), dtype=np.float32)
        resized_cnt_mat = np.zeros((mat_h, mat_w, 1), dtype=np.float32)
        resized_cnt_mat += 1e-12

        for heatMat, pafMat, info in zip(heat_mats, paf_mats, infos):
            w, h = int(info[2] * mat_w), int(info[3] * mat_h)
            heatMat = cv.resize(heatMat, (w, h))
            pafMat = cv.resize(pafMat, (w, h))
            x, y = int(info[0] * mat_w), int(info[1] * mat_h)
            # add up
            resized_heat_mat[max(0, y):y + h, max(0, x):x + w, :] = np.maximum(
                resized_heat_mat[max(0, y):y + h, max(0, x):x + w, :], heatMat[max(0, -y):, max(0, -x):, :])
            resized_paf_mat[max(0, y):y + h, max(0, x):x + w, :] += pafMat[max(0, -y):, max(0, -x):, :]
            resized_cnt_mat[max(0, y):y + h, max(0, x):x + w, :] += 1

        self.heatMat = resized_heat_mat
        self.pafMat = resized_paf_mat / (np.log(resized_cnt_mat) + 1)

        humans = PoseEstimator.estimate(self.heatMat, self.pafMat)
        return humans


class PoseEstimator:
    heatmap_supress = False
    heatmap_gaussian = True
    adaptive_threshold = False

    NMS_Threshold = 0.15
    Local_PAF_Threshold = 0.2
    PAF_Count_Threshold = 5
    Part_Count_Threshold = 4
    Part_Score_Threshold = 4.5

    PartPair = namedtuple('PartPair', ['score', 'part_idx1', 'part_idx2', 'idx1', 'idx2',
                                       'coord1', 'coord2', 'score1', 'score2'], verbose=False)

    @staticmethod
    def non_max_suppression(plain, window_size=3, threshold=NMS_Threshold):
        under_threshold_indices = plain < threshold
        plain[under_threshold_indices] = 0
        return plain * (plain == maximum_filter(plain, footprint=np.ones((window_size, window_size))))

    @staticmethod
    def estimate(heat_mat, paf_mat):
        if heat_mat.shape[2] == 19:
            heat_mat = np.rollaxis(heat_mat, 2, 0)
        if paf_mat.shape[2] == 38:
            paf_mat = np.rollaxis(paf_mat, 2, 0)

        if PoseEstimator.heatmap_supress:
            heat_mat = heat_mat - heat_mat.min(axis=1).min(axis=1).reshape(19, 1, 1)
            heat_mat = heat_mat - heat_mat.min(axis=2).reshape(19, heat_mat.shape[1], 1)

        if PoseEstimator.heatmap_gaussian:
            heat_mat = gaussian_filter(heat_mat, sigma=0.5)

        if PoseEstimator.adaptive_threshold:
            _NMS_Threshold = max(np.average(heat_mat) * 4.0, PoseEstimator.NMS_Threshold)
            _NMS_Threshold = min(_NMS_Threshold, 0.3)
        else:
            _NMS_Threshold = PoseEstimator.NMS_Threshold

        # extract interesting coordinates using NMS.
        coords = []  # [[coords in plane1], [....], ...]
        for plain in heat_mat[:-1]:
            nms = PoseEstimator.non_max_suppression(plain, 5, _NMS_Threshold)
            coords.append(np.where(nms >= _NMS_Threshold))

        # score pairs
        pairs_by_conn = list()
        for (part_idx1, part_idx2), (paf_x_idx, paf_y_idx) in zip(CocoPairs, CocoPairsNetwork):
            pairs = PoseEstimator.score_pairs(
                part_idx1, part_idx2,
                coords[part_idx1], coords[part_idx2],
                paf_mat[paf_x_idx], paf_mat[paf_y_idx],
                heatmap=heat_mat,
                rescale=(1.0 / heat_mat.shape[2], 1.0 / heat_mat.shape[1])
            )

            pairs_by_conn.extend(pairs)

        # merge pairs to human
        # pairs_by_conn is sorted by CocoPairs(part importance) and Score between Parts.
        humans = [Human([pair]) for pair in pairs_by_conn]
        while True:
            merge_items = None
            for k1, k2 in itertools.combinations(humans, 2):
                if k1 == k2:
                    continue
                if k1.is_connected(k2):
                    merge_items = (k1, k2)
                    break

            if merge_items is not None:
                merge_items[0].merge(merge_items[1])
                humans.remove(merge_items[1])
            else:
                break

        # reject by subset count
        humans = [human for human in humans if human.part_count() >= PoseEstimator.PAF_Count_Threshold]
        # reject by subset max score
        humans = [human for human in humans if human.get_max_score() >= PoseEstimator.Part_Score_Threshold]
        return humans

    @staticmethod
    def score_pairs(part_idx1, part_idx2, coord_list1, coord_list2, paf_mat_x, paf_mat_y, heatmap, rescale=(1.0, 1.0)):
        connection_temp = []

        cnt = 0
        for idx1, (y1, x1) in enumerate(zip(coord_list1[0], coord_list1[1])):
            for idx2, (y2, x2) in enumerate(zip(coord_list2[0], coord_list2[1])):
                score, count = PoseEstimator.get_score(x1, y1, x2, y2, paf_mat_x, paf_mat_y)
                cnt += 1
                if count < PoseEstimator.PAF_Count_Threshold or score <= 0.0:
                    continue
                connection_temp.append(PoseEstimator.PartPair(
                    score=score,
                    part_idx1=part_idx1, part_idx2=part_idx2,
                    idx1=idx1, idx2=idx2,
                    coord1=(x1 * rescale[0], y1 * rescale[1]),
                    coord2=(x2 * rescale[0], y2 * rescale[1]),
                    score1=heatmap[part_idx1][y1][x1],
                    score2=heatmap[part_idx2][y2][x2],
                ))

        connection = []
        used_idx1, used_idx2 = set(), set()
        for candidate in sorted(connection_temp, key=lambda x: x.score, reverse=True):
            # check not connected
            if candidate.idx1 in used_idx1 or candidate.idx2 in used_idx2:
                continue
            connection.append(candidate)
            used_idx1.add(candidate.idx1)
            used_idx2.add(candidate.idx2)

        return connection

    @staticmethod
    def get_score(x1, y1, x2, y2, paf_mat_x, paf_mat_y):
        __num_inter = 10
        __num_inter_f = float(__num_inter)
        dx, dy = x2 - x1, y2 - y1
        normVec = math.sqrt(dx ** 2 + dy ** 2)

        if normVec < 1e-4:
            return 0.0, 0

        vx, vy = dx / normVec, dy / normVec

        xs = np.arange(x1, x2, dx / __num_inter_f) if x1 != x2 else np.full((__num_inter,), x1)
        ys = np.arange(y1, y2, dy / __num_inter_f) if y1 != y2 else np.full((__num_inter,), y1)
        xs = (xs + 0.5).astype(np.int8)
        ys = (ys + 0.5).astype(np.int8)

        # without vectorization
        pafXs = np.zeros(__num_inter)
        pafYs = np.zeros(__num_inter)
        for idx, (mx, my) in enumerate(zip(xs, ys)):
            pafXs[idx] = paf_mat_x[my][mx]
            pafYs[idx] = paf_mat_y[my][mx]

        local_scores = pafXs * vx + pafYs * vy
        thidxs = local_scores > PoseEstimator.Local_PAF_Threshold

        return sum(local_scores * thidxs), sum(thidxs)


class CocoPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18


FilePath = Path.cwd()
outFilePath = Path(FilePath / "test_out/")
# 有两种输入：height 368 * width 368 （不保持宽高比）；height 368 * width (保持宽高比)
input_width, input_height = 490, 368

# Usage example:  python openpose.py --video=test.mp4
parser = argparse.ArgumentParser(description='OpenPose for pose skeleton in python')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

skeleton_estimator = None
nPoints = 18
CocoPairs = [(1, 2), (1, 5),  (2, 3),  (3, 4),   (5, 6),   (6, 7), (1, 8),
             (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14),
             (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)]   # = 19
CocoPairsRender = CocoPairs[:-2]
CocoPairsNetwork = [(12, 13), (20, 21), (14, 15), (16, 17), (22, 23), (24, 25), (0, 1),
                    (2, 3),   (4, 5),   (6, 7),   (8, 9),   (10, 11), (28, 29), (30, 31),
                    (34, 35), (32, 33), (36, 37), (18, 19), (26, 27)]  # = 19

# CocoColors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255],
#               [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255],
#               [0, 0, 255],   [255, 0, 0],   [200, 200, 0], [255, 0, 0],   [200, 200, 0], [0, 0, 0]]
CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


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
    global skeleton_estimator
    skeleton_estimator = TfPoseEstimator(
        get_graph_path('VGG_origin'), target_size=(input_width, input_height))


def get_graph_path(model_name):
    dyn_graph_path = {
        'VGG_origin': str(FilePath/"graph_model_coco/graph_opt.pb"),
        'mobilemet': str(FilePath/"graph_model_coco/graph_opt_mobile.pb")
    }
    graph_path = dyn_graph_path[model_name]
    if not os.path.isfile(graph_path):
        raise Exception('Graph file doesn\'t exist, path=%s' % graph_path)
    return graph_path


if __name__ == "__main__":
    cap = choose_run_mode()
    load_pretrain_model()
    fps = FPS().start()
    vid_writer = cv.VideoWriter(outFilePath, cv.VideoWriter_fourcc(*'mp4v'), 15,
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                                 round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    # 画骨架的序列图所需要的白色背景，可更改背景大小
    back_ground_h, back_ground_w = 1500, 2400
    back_ground = np.ones((back_ground_h, back_ground_w), dtype=np.uint8)
    back_ground = cv.cvtColor(back_ground, cv.COLOR_GRAY2BGR)
    back_ground[:, :, :] = 255  # white background
    # 间隔interval帧，抽取骨架信息一次
    interval = 12

    # 在线FPS计算参数
    start_time = time.time()
    fps_interval = 1  # 每隔1秒重新计算帧数
    fps_count = 0
    realtime_fps = 'Starting'

    frame_count = 0
    while cv.waitKey(1) < 0:
        has_frame, frame = cap.read()
        img_copy = np.copy(frame)
        if not has_frame:
            cv.waitKey(3000)
            break

        frame_count += 1
        fps_count += 1
        humans = skeleton_estimator.inference(frame)
        # frame_show = frame
        frame_show = TfPoseEstimator.draw_pose_bbox(frame, humans)[0]

        if frame_count % interval == 0 and frame_count <= 180:
            TfPoseEstimator.draw_pose_sequence_horizontally(frame, humans, frame_count)

        # FPS的实时显示
        fps_show = 'FPS:{0:.4}'.format(realtime_fps)
        cv.putText(frame, fps_show, (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if (time.time() - start_time) > fps_interval:
            # 计算这个interval过程中的帧数，若interval为1秒，则为FPS
            realtime_fps = fps_count / (time.time() - start_time)
            fps_count = 0  # 帧数清零
            start_time = time.time()

        winName = 'Pose Skeleton from OpenPose'
        cv.imshow(winName, frame_show)

        # Write the frame with the detection boxes
        if args.image:
            cv.imwrite(outFilePath, frame_show)
        else:
            # vid_writer.write(frame_show)
            vid_writer.write(img_copy)

    if not args.image:
        fps.stop()
        vid_writer.release()
        cap.release()
        cv.imwrite('skeleton_sequence.jpg', back_ground)
    cv.destroyAllWindows()
