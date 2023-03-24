import cv2
import os
import mediapipe as mp
import numpy as np
import pandas as pd
import winsound
from ffpyplayer.player import MediaPlayer
import time
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from mpl_toolkits import mplot3d
# from celluloid import Camera
from scipy import spatial
import pyshine as ps

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 定义计算角度的函数
def calculateAngle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# cap = cv2.VideoCapture(0)
# 2D
def extractKeypoint(path):
    IMAGE_FILES = [path]  # 文件图像

    stage = None  # 即不会为整个Pipeline运行分配全局代理，并且每个stage部分将需要包含其自己的agent部分
    joint_list_video = pd.DataFrame([])
    """
    DataFrame有行和列的索引；它可以被看作是一个Series的字典（Series们共享一个索引）
        state	year	pop
    0	Ohio	2000.0	1.5
    1	Ohio	2001.0	1.7
    2	Ohio	2002.0	3.6
    3	Nevada	2001.0	2.4
    4	Nevada	NaN	2.9
    """
    count = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:  # 默认0.5， 0.5
        # 检测置信度大于0.5代表检测到了，若此时跟踪置信度大于0.5就继续跟踪，小于就沿用上一次，避免一次又一次重复使用模型
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR转RGB通道用于pose加工
            image.flags.writeable = False  # 加工前，flag为False
            results = pose.process(image)  # 对图片进行pose_Detect

            image.flags.writeable = True  # 加工后，flag为True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB转RGB通道用于尺寸检测
            image_h, image_w, _ = image.shape
            # 尺寸录入
            try:

                landmarks = results.pose_landmarks.landmark  # 获取所有关键点信息


                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                joints = []
                joint_list = pd.DataFrame([])
                # zip意思是并行遍历，就是i作为index，每次遍历将每个joint点可视化出来， i = 32退出循环
                for i, data_point in zip(range(len(landmarks)), landmarks):
                    joints = pd.DataFrame({  # 记录到pandas的字典集中
                        'frame': count,
                        'id': i,
                        'x': data_point.x,
                        'y': data_point.y,
                        'z': data_point.z,
                        'vis': data_point.visibility
                    }, index=[0])
                    joint_list = joint_list.append(joints, ignore_index=True)
                    # 将读取字典集加入到新的joint_list字典集中
                    # 将landmarks各个可视的点读入关键点中,33个关键点
                keypoints = []
                for point in landmarks:
                    keypoints.append({
                        'X': point.x,
                        'Y': point.y,
                        'Z': point.z,
                    })

                angle = []
                angle_list = pd.DataFrame([])
                # 角度字典集
                # 对于右肘，右肩膀，右腕的角度计算,后加入到angle列表中
                angle1 = calculateAngle(right_shoulder, right_elbow, right_wrist)
                angle.append(int(angle1))
                # 对于左肘，左肩膀，左腕的角度计算,后加入到angle列表中
                angle2 = calculateAngle(left_shoulder, left_elbow, left_wrist)
                angle.append(int(angle2))
                # 对于右肘，右肩膀，右臀部，髋部的角度计算,后加入到angle列表中
                angle3 = calculateAngle(right_elbow, right_shoulder, right_hip)
                angle.append(int(angle3))
                # 对于左肘，左肩膀，左臀部，髋部的角度计算,后加入到angle列表中
                angle4 = calculateAngle(left_elbow, left_shoulder, left_hip)
                angle.append(int(angle4))
                # 对于右肩膀，右髋部，臀部，右膝盖的角度计算,后加入到angle列表中
                angle5 = calculateAngle(right_shoulder, right_hip, right_knee)
                angle.append(int(angle5))
                # 对于左肩膀，左臀部，左膝盖的角度计算,后加入到angle列表中
                angle6 = calculateAngle(left_shoulder, left_hip, left_knee)
                angle.append(int(angle6))
                # 对于右臀部，右膝盖，右脚腕的角度计算, 后加入到angle列表中
                angle7 = calculateAngle(right_hip, right_knee, right_ankle)
                angle.append(int(angle7))
                # 对于左臀部，左膝盖，左脚腕的角度计算, 后加入到angle列表中
                angle8 = calculateAngle(left_hip, left_knee, left_ankle)
                angle.append(int(angle8))
                # 分别对于左右手腕，肩膀，臀部，膝盖进行说明
                # cv.putText()参数说明
                # 图片
                # 要添加的文字
                # 文字添加到图片上的位置
                # 字体的类型
                # 字体大小
                # 字体颜色
                # 字体粗细
                cv2.putText(image,
                            str(1),
                            tuple(np.multiply(right_elbow, [image_w, image_h, ]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            [255, 255, 0],
                            2,
                            cv2.LINE_AA
                            )
                cv2.putText(image,
                            str(2),
                            tuple(np.multiply(left_elbow, [image_w, image_h]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            [255, 255, 0],
                            2,
                            cv2.LINE_AA
                            )
                cv2.putText(image,
                            str(3),
                            tuple(np.multiply(right_shoulder, [image_w, image_h]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            [255, 255, 0],
                            2,
                            cv2.LINE_AA
                            )
                cv2.putText(image,
                            str(4),
                            tuple(np.multiply(left_shoulder, [image_w, image_h]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            [255, 255, 0],
                            2,
                            cv2.LINE_AA
                            )
                cv2.putText(image,
                            str(5),
                            tuple(np.multiply(right_hip, [image_w, image_h]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            [255, 255, 0],
                            2,
                            cv2.LINE_AA
                            )
                cv2.putText(image,
                            str(6),
                            tuple(np.multiply(left_hip, [image_w, image_h]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            [255, 255, 0],
                            2,
                            cv2.LINE_AA
                            )
                cv2.putText(image,
                            str(7),
                            tuple(np.multiply(right_knee, [image_w, image_h]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            [255, 255, 0],
                            2,
                            cv2.LINE_AA
                            )
                cv2.putText(image,
                            str(8),
                            tuple(np.multiply(left_knee, [image_w, image_h]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            [255, 255, 0],
                            2,
                            cv2.LINE_AA
                            )



            #             if angle >120:
            #                 stage = "down"
            #             if angle <30 and stage == 'down':
            #                 stage = "up"
            #                 counter +=1

            # 定义 try 块引发错误时要运行的代码块
            except:
                pass 


            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
    return landmarks, keypoints, angle, image



def compare_pose(image, angle_point, angle_user, angle_target):
            angle_user = np.array(angle_user)
            angle_target = np.array(angle_target)
            angle_point = np.array(angle_point)
            # stage = 0
            stage = [0] * 16
            # cv2.rectangle(image, (0, 0), (370, 40), (255, 255, 255), -1)
            # cv2.rectangle(image, (0, 40), (370, 370), (255, 255, 255), -1)
            cv2.putText(image, str("Score:"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
            height, width, _ = image.shape

            if angle_user[0] < (angle_target[0] - 15):
                # print("Extend the right arm at elbow")
                # stage = stage + 1
                stage[0] = 1
                cv2.putText(image, str("Extend the right arm at elbow"), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            [0, 153, 0], 2, cv2.LINE_AA)
                cv2.circle(image, (int(angle_point[0][0] * width), int(angle_point[0][1] * height)), 30, (0, 0, 255), 5)

            if angle_user[0] > (angle_target[0] + 15):
                # print("Fold the right arm at elbow")
                # stage = stage + 1
                stage[1] = 1
                cv2.putText(image, str("Fold the right arm at elbow"), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            [0, 153, 0], 2, cv2.LINE_AA)
                cv2.circle(image, (int(angle_point[0][0] * width), int(angle_point[0][1] * height)), 30, (0, 0, 255), 5)

            if angle_user[1] < (angle_target[1] - 15):
                # print("Extend the left arm at elbow")
                # stage = stage + 1
                stage[2] = 1
                cv2.putText(image, str("Extend the left arm at elbow"), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            [0, 153, 0], 2, cv2.LINE_AA)
                cv2.circle(image, (int(angle_point[1][0] * width), int(angle_point[1][1] * height)), 30, (0, 0, 255), 5)

            if angle_user[1] > (angle_target[1] + 15):
                # print("Fold the left arm at elbow")
                # stage = stage + 1
                stage[3] = 1
                cv2.putText(image, str("Fold the left arm at elbow"), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            [0, 153, 0], 2, cv2.LINE_AA)
                cv2.circle(image, (int(angle_point[1][0] * width), int(angle_point[1][1] * height)), 30, (0, 0, 255), 5)

            if angle_user[2] < (angle_target[2] - 15):
                # print("Lift your right arm")
                # stage = stage + 1
                stage[4] = 1
                cv2.putText(image, str("Lift your right arm"), (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2,
                            cv2.LINE_AA)
                cv2.circle(image, (int(angle_point[2][0] * width), int(angle_point[2][1] * height)), 30, (0, 0, 255), 5)

            if angle_user[2] > (angle_target[2] + 15):
                # print("Put your arm down a little")
                # stage = stage + 1
                stage[5] = 1
                cv2.putText(image, str("Put your arm down a little"), (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            [0, 153, 0], 2, cv2.LINE_AA)
                cv2.circle(image, (int(angle_point[2][0] * width), int(angle_point[2][1] * height)), 30, (0, 0, 255), 5)

            if angle_user[3] < (angle_target[3] - 15):
                # print("Lift your left arm")
                # stage = stage + 1
                stage[6] = 1
                cv2.putText(image, str("Lift your left arm"), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2,
                            cv2.LINE_AA)
                cv2.circle(image, (int(angle_point[3][0] * width), int(angle_point[3][1] * height)), 30, (0, 0, 255), 5)

            if angle_user[3] > (angle_target[3] + 15):
                # print("Put your arm down a little")
                # stage = stage + 1
                stage[7] = 1
                cv2.putText(image, str("Put your arm down a little"), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            [0, 153, 0], 2, cv2.LINE_AA)
                cv2.circle(image, (int(angle_point[3][0] * width), int(angle_point[3][1] * height)), 30, (0, 0, 255), 5)

            if angle_user[4] < (angle_target[4] - 15):
                # print("Extend the angle at right hip")
                # stage = stage + 1
                stage[8] = 1
                cv2.putText(image, str("Extend the angle at right hip"), (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            [0, 153, 0], 2, cv2.LINE_AA)
                cv2.circle(image, (int(angle_point[4][0] * width), int(angle_point[4][1] * height)), 30, (0, 0, 255), 5)

            if angle_user[4] > (angle_target[4] + 15):
                # print("Reduce the angle at right hip")
                # stage = stage + 1
                stage[9] = 1
                cv2.putText(image, str("Reduce the angle of at right hip"), (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            [0, 153, 0], 2, cv2.LINE_AA)
                cv2.circle(image, (int(angle_point[4][0] * width), int(angle_point[4][1] * height)), 30, (0, 0, 255), 5)

            if angle_user[5] < (angle_target[5] - 15):
                # print("Extend the angle at left hip")
                # stage = stage + 1
                stage[10] = 1
                cv2.putText(image, str("Extend the angle at left hip"), (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            [0, 153, 0], 2, cv2.LINE_AA)
                cv2.circle(image, (int(angle_point[5][0] * width), int(angle_point[5][1] * height)), 30, (0, 0, 255), 5)

            if angle_user[5] > (angle_target[5] + 15):
                # print("Reduce the angle at left hip")
                # stage = stage + 1
                stage[11] = 1
                cv2.putText(image, str("Reduce the angle at left hip"), (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            [0, 153, 0], 2, cv2.LINE_AA)
                cv2.circle(image, (int(angle_point[5][0] * width), int(angle_point[5][1] * height)), 30, (0, 0, 255), 5)

            if angle_user[6] < (angle_target[6] - 15):
                # print("Extend the angle of right knee")
                # stage = stage + 1
                stage[12] = 1
                cv2.putText(image, str("Extend the angle of right knee"), (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            [0, 153, 0], 2, cv2.LINE_AA)
                cv2.circle(image, (int(angle_point[6][0] * width), int(angle_point[6][1] * height)), 30, (0, 0, 255), 5)

            if angle_user[6] > (angle_target[6] + 15):
                # print("Reduce the angle of right knee")
                # stage = stage + 1
                stage[13] = 1
                cv2.putText(image, str("Reduce the angle at right knee"), (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            [0, 153, 0], 2, cv2.LINE_AA)
                cv2.circle(image, (int(angle_point[6][0] * width), int(angle_point[6][1] * height)), 30, (0, 0, 255), 5)

            if angle_user[7] < (angle_target[7] - 15):
                # print("Extend the angle at left knee")
                # stage = stage + 1
                stage[14] = 1
                cv2.putText(image, str("Extend the angle at left knee"), (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            [0, 153, 0], 2, cv2.LINE_AA)
                cv2.circle(image, (int(angle_point[7][0] * width), int(angle_point[7][1] * height)), 30, (0, 0, 255), 5)

            if angle_user[7] > (angle_target[7] + 15):
                # print("Reduce the angle at left knee")
                # stage = stage + 1
                stage[15] = 1
                cv2.putText(image, str("Reduce the angle at left knee"), (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            [0, 153, 0], 2, cv2.LINE_AA)
                cv2.circle(image, (int(angle_point[7][0] * width), int(angle_point[7][1] * height)), 30, (0, 0, 255), 5)

            add_stage = sum(stage)  # stage列表求和
            if add_stage == 0:
                cv2.putText(image, str("PERFECT"), (170, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2, cv2.LINE_AA)
            elif add_stage < 4:
                cv2.putText(image, str("FIGHTING!"), (170, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2, cv2.LINE_AA)
            else:
                cv2.putText(image, str("WARNING!"), (170, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2, cv2.LINE_AA)
            return stage, add_stage

def Average(lst):
     return sum(lst) / len(lst)

def dif_compare(x,y):
    average = []
    for i, j in zip(range(len(list(x))),range(len(list(y)))):
        result = 1 - spatial.distance.cosine(list(x[i].values()),list(y[j].values()))
        average.append(result)
    score = math.sqrt(2*(1-round(Average(average),2)))
    #print(Average(average))
    return score

def diff_compare_angle(x,y):
    new_x = []
    for i, j in zip(range(len(x)), range(len(y))):
        z = np.abs(x[i] - y[j])/((x[i] + y[j])/2)
        new_x.append(z)
        #print(new_x[i])
    return Average(new_x)

def convert_data(landmarks):
    df = pd.DataFrame(columns=['x', 'y', 'z', 'vis'])
    for i in range(len(landmarks)):
        df = df.append({"x": landmarks[i].x,
                        "y": landmarks[i].y,
                        "z": landmarks[i].z,
                        "vis": landmarks[i].visibility
                        }, ignore_index=True)
    return df



cap = cv2.VideoCapture(0)  # 捕获摄像头设备
path = []
path += ["pic/test.jpg"]  # "Video/yoga19.jpg"  # 标准健身图片路径
path_static = "static_pic"  # 该处填写一批静态图所在的文件夹名
x = extractKeypoint(path[0])  # 进入extractKeypoint函数：提取关键点
dim = (960, 760)
resized = cv2.resize(x[3], dim, interpolation=cv2.INTER_AREA)
# cv2.imshow('target', resized)
angle_target = x[2]
point_target = x[1]
num = 0  # 保存图片的张数，初始为0
IMAGE_FILES = [path_static]
# 音频部分
# duration = 1000  # millisecond
# freq = 440  # Hz

video_path = "./test.mp3"
audio_dict = {1: "./1_alert.mp3", 2: "./2_alert.mp3", 3: "./3_alert.mp3"}

# 记录次数
record = [0] * 16

with mp_pose.Pose(static_image_mode=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    for filename in os.listdir(path_static):
        ret = True
        image = cv2.imread(path_static + '/' + filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape
        image = cv2.resize(image, (int(image_width * (860 / image_height)), 860))
        # finding the distance by calling function
        # Distance distance finder function need
        # these arguments the Focal_Length,
        # Known_width(centimeters),
        # and Known_distance(centimeters)

        try:
            landmarks = results.pose_landmarks.landmark

            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z,
                        round(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility * 100, 2)]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z,
                     round(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility * 100, 2)]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z,
                     round(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility * 100, 2)]

            angle_point = []

            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            angle_point.append(right_elbow)

            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            angle_point.append(left_elbow)

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            angle_point.append(right_shoulder)

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            angle_point.append(left_shoulder)

            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            angle_point.append(right_hip)

            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            angle_point.append(left_hip)

            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            angle_point.append(right_knee)

            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            angle_point.append(left_knee)
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            keypoints = []
            for point in landmarks:
                keypoints.append({
                    'X': point.x,
                    'Y': point.y,
                    'Z': point.z,
                })

            p_score = dif_compare(keypoints, point_target)

            angle = []

            angle1 = calculateAngle(right_shoulder, right_elbow, right_wrist)
            angle.append(int(angle1))
            angle2 = calculateAngle(left_shoulder, left_elbow, left_wrist)
            angle.append(int(angle2))
            angle3 = calculateAngle(right_elbow, right_shoulder, right_hip)
            angle.append(int(angle3))
            angle4 = calculateAngle(left_elbow, left_shoulder, left_hip)
            angle.append(int(angle4))
            angle5 = calculateAngle(right_shoulder, right_hip, right_knee)
            angle.append(int(angle5))
            angle6 = calculateAngle(left_shoulder, left_hip, left_knee)
            angle.append(int(angle6))
            angle7 = calculateAngle(right_hip, right_knee, right_ankle)
            angle.append(int(angle7))
            angle8 = calculateAngle(left_hip, left_knee, left_ankle)
            angle.append(int(angle8))

            stage = compare_pose(image, angle_point, angle, angle_target)  # 返回stage[0]: stage列表 和 stage[1]: sum(stage)

            a_score = diff_compare_angle(angle, angle_target)

            if p_score >= a_score:
                cv2.putText(image, str(int((1 - a_score) * 100)), (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2,
                            cv2.LINE_AA)

            else:
                cv2.putText(image, str(int((1 - p_score) * 100)), (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2,
                            cv2.LINE_AA)

        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3)
                                  )
        cv2.namedWindow('AI Exercise', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow('target', resized)
        cv2.imshow('AI Exercise', image)
        # cv2.imwrite(filename, image)
        num += 1
        Img_Name = "./saved_pic/" + str(num) + ".jpg"  # 储存到指定文件夹
        cv2.imwrite(Img_Name, image)

        # if stage[1] >= 4:  # 预警提示，后续可针对不同部位进行提示
            # winsound.Beep(freq, duration)
        if cv2.waitKey(10000) & 0xFF == 27:  # 按Esc退出
            break

    # cap.release()
    cv2.destroyAllWindows()

# camera mode
# player = MediaPlayer(video_path)
cv2.imshow('target', resized)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():

        """
        audio_frame, val = player.get_frame()
        if val != 'eof' and audio_frame is not None:
            # audio
            img, t = audio_frame
            # print(img, t)
        """
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)  # 水平翻转
        if not success:
            print("Something error with your camera.")
            # If loading a video, use 'break' instead of 'continue'.
            # but right now we compare to images, maybe later can support video
            continue

        # read frame image from real-time camera
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # set flag non-writeable and record the result arrays
        image.flags.writeable = False
        results = pose.process(image)

        # if we want to draw landmarks we need to make it writeable
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # shape is a tuple of three values (height, width, channels) ,_ is hold for the channel,
        image_height, image_width, _ = image.shape
        image = cv2.resize(image, (int(image_width * (860 / image_height)), 860))

        try:

            # get the result landmark and angle_point
            landmarks = results.pose_landmarks.landmark
            angle_point = []

            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z,
                        round(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility * 100, 2)]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z,
                     round(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility * 100, 2)]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z,
                     round(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility * 100, 2)]
            # 下面把frame中的各个部位关键点读取到angle_point列表中
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            angle_point.append(right_elbow)

            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            angle_point.append(left_elbow)

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            angle_point.append(right_shoulder)

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            angle_point.append(left_shoulder)

            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            angle_point.append(right_hip)

            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            angle_point.append(left_hip)

            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            angle_point.append(right_knee)

            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            angle_point.append(left_knee)
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            keypoints = []
            for point in landmarks:
                keypoints.append({
                    'X': point.x,
                    'Y': point.y,
                    'Z': point.z,
                })

            # compare key points with target points
            p_score = dif_compare(keypoints, point_target)

            angle = []

            angle1 = calculateAngle(right_shoulder, right_elbow, right_wrist)
            angle.append(int(angle1))
            angle2 = calculateAngle(left_shoulder, left_elbow, left_wrist)
            angle.append(int(angle2))
            angle3 = calculateAngle(right_elbow, right_shoulder, right_hip)
            angle.append(int(angle3))
            angle4 = calculateAngle(left_elbow, left_shoulder, left_hip)
            angle.append(int(angle4))
            angle5 = calculateAngle(right_shoulder, right_hip, right_knee)
            angle.append(int(angle5))
            angle6 = calculateAngle(left_shoulder, left_hip, left_knee)
            angle.append(int(angle6))
            angle7 = calculateAngle(right_hip, right_knee, right_ankle)
            angle.append(int(angle7))
            angle8 = calculateAngle(left_hip, left_knee, left_ankle)
            angle.append(int(angle8))

            stage = compare_pose(image, angle_point, angle, angle_target)  # 与上同理

            a_score = diff_compare_angle(angle, angle_target)

            if p_score >= a_score:
                cv2.putText(image, str(int((1 - a_score) * 100)), (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            [0, 0, 255], 2,
                            cv2.LINE_AA)
            else:
                cv2.putText(image, str(int((1 - p_score) * 100)), (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            [0, 0, 255], 2,
                            cv2.LINE_AA)

        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3)
                                  )
        cv2.imshow('AI Exercise', image)

        for index, val in enumerate(stage[0]):
            if index + 1 <= 3 and val == 1:
                record[index + 1] += 1
                if record[index + 1] >= 200:
                    record[index + 1] = 0
                    player = MediaPlayer(audio_dict[index + 1])



        if cv2.waitKey(10) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            break



