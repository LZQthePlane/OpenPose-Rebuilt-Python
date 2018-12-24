import os
import sys
import cv2 as cv
import numpy as np
from pathlib import Path
import argparse
from imutils.video import FPS


# reference: https://blog.csdn.net/qq_27158179/article/details/82717821

# 相关参数初始化
FilePath = Path.cwd()
out_file_path = Path(FilePath / "test_out/")

joints_list_with_id = []  # 储存frame中所有joints, [[（x, y, conf, id), ...], [（x, y, conf, id), ...]]
joints_list = np.zeros((0, 3))  # 储存frame中所有joints, [[x, y, conf], ...]
joint_id = 0  # frame中所有joints的id

# 以COCO骨架格式为例
proto_file = str(FilePath / "model/coco/pose_deploy_linevec_faster_4_stages.prototxt")
weights_file = str(FilePath / "model/coco/pose_iter_440000.caffemodel")

coco_num = 18
joints_mapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr',
                  'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye',
                  'R-Ear', 'L-Ear', 'Background']
POSE_PAIRS = [[1, 2], [1, 5],  [2, 3],   [3, 4],  [5, 6],   [6, 7],
              [1, 8], [8, 9],  [9, 10],  [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]
# index of PAFs corresponding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
MAP_INDEX = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
             [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
             [47, 48], [49, 50], [53, 54], [51, 52], [55, 56], [37, 38], [45, 46]]

COLORS = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255],
          [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255],
          [0, 0, 255],   [255, 0, 0],   [200, 200, 0], [255, 0, 0],   [200, 200, 0], [0, 0, 0]]


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


def get_joints(prob_map, threshold=0.15):
    """
    :param prob_map: image中某一类joint的map
    :param threshold: 以这个threshold过滤掉map中置信度较低的部分
    :return: 所有该joint类别的集合，形式为(x, y, conf)
    """
    # 以较低的threshold值，提取出可能为某关节点的所有像素区域
    map_smooth = cv.GaussianBlur(prob_map, (3, 3), 0, 0)
    # 生成这个区域的mask
    map_mask = np.uint8(map_smooth > threshold)
    joints = []
    # 围绕mask画轮廓contour
    _, contours, _ = cv.findContours(map_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # 针对每一个person某joint的contour，找到confidence最大的一个点，作为估计的关节点位置
    for cnt in contours:
        blob_mask = np.zeros(map_mask.shape)
        blob_mask = cv.fillConvexPoly(blob_mask, cnt, 1)
        masked_prob_map = map_smooth * blob_mask
        _, max_val, _, max_loc = cv.minMaxLoc(masked_prob_map)
        joints.append(max_loc + (prob_map[max_loc[1], max_loc[0]],))
        # 返回
    return joints


def get_valid_pairs(output):
    """
    Find valid / invalid connections between the different joints of a all persons present
    """
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10  # 插值点数目
    paf_threshold = 0.2
    conf_threshold = 0.5
    # loop for every POSE_PAIR
    for k in range(len(MAP_INDEX)):
        # a->b constitute a limb
        paf_a = output[0, MAP_INDEX[k][0], :, :]
        # print(paf_a.shape)
        paf_b = output[0, MAP_INDEX[k][1], :, :]
        paf_a = cv.resize(paf_a, (frameWidth, frameHeight))
        paf_b = cv.resize(paf_b, (frameWidth, frameHeight))

        # Find the joints for the first and second limb
        # cand_a为某一joint的列表， cand_b为另一与之相连接的joint的列表
        cand_a = joints_list_with_id[POSE_PAIRS[k][0]]
        cand_b = joints_list_with_id[POSE_PAIRS[k][1]]
        # 在完美检测到frame中所有joints的情况下， n_a = n_b = len(persons)
        n_a = len(cand_a)
        n_b = len(cand_b)

        # If joints for the joint-pair is detected
        # check every joint in cand_a with every joint in cand_b
        if n_a != 0 and n_b != 0:
            valid_pair = np.zeros((0, 3))
            for i in range(n_a):
                max_j = -1
                max_score = -1
                found = False
                for j in range(n_b):
                    # Calculate the distance vector between the two joints
                    distance_ij = np.subtract(cand_b[j][:2], cand_a[i][:2])
                    # 求二范数，即求模，算两点距离
                    norm = np.linalg.norm(distance_ij)
                    if norm:
                        # 距离不为零的话， 缩放到单位向量
                        distance_ij = distance_ij / norm
                    else:
                        continue

                    # Find p(u)，在连接两joints的直线上创建一个n_interp_samples插值点的数组
                    interp_coord = list(zip(np.linspace(cand_a[i][0], cand_b[j][0], num=n_interp_samples),
                                            np.linspace(cand_a[i][1], cand_b[j][1], num=n_interp_samples)))
                    # Find the PAF values at a set of interpolated points between the joints
                    paf_interp = []
                    for m in range(len(interp_coord)):
                        paf_interp.append([paf_a[int(round(interp_coord[m][1])), int(round(interp_coord[m][0]))],
                                           paf_b[int(round(interp_coord[m][1])), int(round(interp_coord[m][0]))]])
                    # Find E
                    paf_scores = np.dot(paf_interp, distance_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if (len(np.where(paf_scores > paf_threshold)[0]) / n_interp_samples) > conf_threshold:
                        if avg_paf_score > max_score:
                            # 如果这些点中有70%大于conf threshold，则把这一对当成有效
                            max_j = j
                            max_score = avg_paf_score
                            found = True
                # Append the connection to the list
                if found:
                    valid_pair = np.append(valid_pair, [[cand_a[i][3], cand_b[max_j][3], max_score]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        # If no joints are detected
        else:
            # print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs


def get_personwise_joints(valid_pairs, invalid_pairs):
    """
    This function creates a list of joints belonging to each person
    For each detected valid pair, it assigns the joint(s) to a person
    """
    # the last number in each row is the overall score
    personwise_joints = -1 * np.ones((0, 19))
    # print(personwise_joints.shape)

    for k in range(len(MAP_INDEX)):
        if k not in invalid_pairs:
            part_as = valid_pairs[k][:, 0]
            part_bs = valid_pairs[k][:, 1]
            index_a, index_b = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = False
                person_idx = -1
                for j in range(len(personwise_joints)):
                    if personwise_joints[j][index_a] == part_as[i]:
                        person_idx = j
                        found = True
                        break

                if found:
                    personwise_joints[person_idx][index_b] = part_bs[i]
                    personwise_joints[person_idx][-1] += joints_list[part_bs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[index_a] = part_as[i]
                    row[index_b] = part_bs[i]
                    # add the joint_scores for the two joints and the paf_score
                    row[-1] = sum(joints_list[valid_pairs[k][i, :2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwise_joints = np.vstack([personwise_joints, row])
    return personwise_joints


def load_pretrained_model():
    """
    加载预训练后的POSE的caffe模型
    """
    net = cv.dnn.readNetFromCaffe(proto_file, weights_file)
    # 调用GPU模块，但目前opencv仅支持intel GPU
    # tested with Intel GPUs only, or it will automatically switch to CPU
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)
    return net


def show_joints(frame):
    for i in range(coco_num):
        for j in range(len(joints_list_with_id[i])):
            cv.circle(frame, joints_list_with_id[i][j][0:2], 5, COLORS[i], -1, cv.LINE_AA)
    cv.imshow("joints", frame)


if __name__ == '__main__':
    # Usage example:  python estimator_single_person.py --video=test.mp4
    parser = argparse.ArgumentParser(description='OpenPose for pose skeleton in python')
    parser.add_argument('--image', help='Path to image file.')
    parser.add_argument('--video', help='Path to video file.')
    args = parser.parse_args()

    cap = choose_run_mode()
    net = load_pretrained_model()
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

        frameHeight, frameWidth = frame.shape[:2]

        # Fix the input Height and get the width according to the Aspect Ratio
        # 有两种输入：height 368 * width 368 （不保持宽高比）；height 368 * width (保持宽高比)
        inHeight, inwidth = 368, 368
        inWidth = int((inHeight/frameHeight)*frameWidth)  # 源代码中是选择 保持宽高比

        # 首先，我们将像素值标准化为（0,1）。然后我们指定图像的尺寸。接下来，要减去的平均值，即（0,0,0）
        # 不同算法及训练模型的blobFromImage参数不同，可访问opencv的github地址查询
        # https://github.com/opencv/opencv/tree/master/samples/dnn
        inpBlob = cv.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        # output的[0, i, :, :,]的前19个矩阵为19个（包含一个background）关节点的置信map
        # 后38个矩阵为每两个相连接关节点之间形成的PAF矩阵
        output = net.forward()

        # 找到image中所有可能的关节点joint的坐标
        for part in range(coco_num):
            # 按关节点的序列分别寻找
            joint_prob_map = output[0, part, :, :]
            joint_prob_map = cv.resize(joint_prob_map, (frame.shape[1], frame.shape[0]))
            # 调用get_joints函数，获取image中的所有人的某一joint，len(joints)==len(persons)
            joints = get_joints(joint_prob_map)

            joints_with_id = []
            for i in range(len(joints)):
                # joints集合，竖直叠加（无id）
                joints_list = np.vstack([joints_list, joints[i]])
                # joints集合（有id）
                joints_with_id.append(joints[i] + (joint_id,))
                joint_id += 1
            joints_list_with_id.append(joints_with_id)

        # # 标出所有joints位置(no joints connection)
        # show_joints(frame.copy())

        # 画多人skeleton（joints connection）
        valid_pairs, invalid_pairs = get_valid_pairs(output)
        personwise_joints = get_personwise_joints(valid_pairs, invalid_pairs)

        for i in range(17):
            for n in range(len(personwise_joints)):
                index = personwise_joints[n][np.array(POSE_PAIRS[i])]
                if -1 in index:
                    continue
                B = np.int32(joints_list[index.astype(int), 0])
                A = np.int32(joints_list[index.astype(int), 1])
                cv.line(frame, (B[0], A[0]), (B[1], A[1]), COLORS[i], 3, cv.LINE_AA)

        # Write the frame with the detection boxes
        if args.image:
            cv.imwrite(out_file_path, frame)
        else:
            vid_writer.write(frame)

        cv.imshow("Detected Pose", frame)

    if not args.image:
        vid_writer.release()
        cap.release()
    cv.destroyAllWindows()
