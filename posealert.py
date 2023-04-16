import time
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
from scipy import spatial
import threading
import queue
import pyttsx3
from moviepy.editor import VideoFileClip
from PIL import Image, ImageDraw, ImageFont
import numpy



mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
capqueue = queue.Queue()
tqueue = queue.Queue()
p_score = 0
clip_time = 0
# path = ""
isEnd = False  # 按一次q，则触发窗口全退出

class videocompare:
    def __init__(self, targetpath, H, W_1, W_2, W_3, W):
        global isEnd
        isEnd = False
        # global path
        # read video time
        global clip_time
        clip = VideoFileClip(targetpath)
        clip_time = clip.duration
        # global path
        # path = targetpath
        # Main function starts here
        cap = cv2.VideoCapture(0)
        target_cap = cv2.VideoCapture(targetpath)

        # Create threads for each function
        target_thread = threading.Thread(target=detect_video, args=(target_cap, tqueue, H, W_2, W))
        camera_thread = threading.Thread(target=detect_pose, args=(cap, capqueue, H, W_3, W))

        target_thread.start()
        camera_thread.start()

        # Compare thread can start
        compare_thread = threading.Thread(target=video_compare_pose, args=(capqueue, tqueue, H, W_1))
        compare_thread.start()
        voice_thread = threading.Thread(target=voice_alert)
        voice_thread.start()

        camera_thread.join()
        target_thread.join()
        compare_thread.join()
        voice_thread.join()

        cap.release()
        target_cap.release()
        cv2.destroyAllWindows()
        # exit()
        tqueue.queue.clear()
        capqueue.queue.clear()


class posecompare:
    def __init__(self, targetpath, H, W_2, W_3, S_w):
        cap = cv2.VideoCapture(0)

        x = extractKeypoint(targetpath)
        angle_target = x[2]
        point_target = x[1]
        image_target = x[3]


        # target window screen height and width
        dim = (814, 610)
        resized = cv2.resize(image_target, dim, interpolation=cv2.INTER_AREA)
        # cv2.imshow('target', resized)

        cv2.namedWindow('target', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('target', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow('target', int(0.45 * S_w), int(0.337 * S_w))
        cv2.moveWindow('target', W_2, int(H))
        cv2.imshow('target', resized)

        # start the real-time pose detection and compare to the target
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                success, frame = cap.read()
                frame = cv2.flip(frame, 1)

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

                    image = compare_pose(image, angle_point, angle, angle_target)

                    a_score = diff_compare_angle(angle, angle_target)

                    if (p_score >= a_score):
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

                cv2.namedWindow('AI Exercise', cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty('AI Exercise', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                # cv2.namedWindow('AI Exercise', 0)
                # cv2.resizeWindow('AI Exercise', 814, 644)
                cv2.resizeWindow('AI Exercise', int(0.45 * S_w), int(0.356 * S_w))
                # cv2.moveWindow('AI Exercise', 1104, 197)
                cv2.moveWindow('AI Exercise', W_3, int(H))
                cv2.imshow('AI Exercise', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break


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
    IMAGE_FILES = [path]
    joint_list_video = pd.DataFrame([])
    count = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_h, image_w, _ = image.shape

            try:

                landmarks = results.pose_landmarks.landmark

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

                joint_list = pd.DataFrame([])

                for i, data_point in zip(range(len(landmarks)), landmarks):
                    joints = pd.DataFrame({
                        'frame': count,
                        'id': i,
                        'x': data_point.x,
                        'y': data_point.y,
                        'z': data_point.z,
                        'vis': data_point.visibility
                    }, index=[0])
                    joint_list = pd.concat([joint_list, joints], ignore_index=True)

                keypoints = []
                for point in landmarks:
                    keypoints.append({
                        'X': point.x,
                        'Y': point.y,
                        'Z': point.z,
                    })

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

            except:
                pass

            joint_list_video = pd.concat([joint_list_video, joint_list], ignore_index=True)

            cv2.rectangle(image, (0, 0), (100, 255), (255, 255, 255), -1)

            cv2.putText(image, 'ID', (10, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 255], 2, cv2.LINE_AA)
            cv2.putText(image, str(1), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
            cv2.putText(image, str(2), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
            cv2.putText(image, str(3), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
            cv2.putText(image, str(4), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
            cv2.putText(image, str(5), (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
            cv2.putText(image, str(6), (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
            cv2.putText(image, str(7), (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
            cv2.putText(image, str(8), (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)

            cv2.putText(image, 'Angle', (40, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 255], 2, cv2.LINE_AA)
            cv2.putText(image, str(int(angle1)), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
            cv2.putText(image, str(int(angle2)), (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
            cv2.putText(image, str(int(angle3)), (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
            cv2.putText(image, str(int(angle4)), (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
            cv2.putText(image, str(int(angle5)), (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
            cv2.putText(image, str(int(angle6)), (40, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
            cv2.putText(image, str(int(angle7)), (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
            cv2.putText(image, str(int(angle8)), (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=2))

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
    return landmarks, keypoints, angle, image


def compare_pose(image, angle_point, angle_user, angle_target):
    angle_user = np.array(angle_user)
    angle_target = np.array(angle_target)
    angle_point = np.array(angle_point)
    stage = 0
    # cv2.rectangle(image, (0, 0), (370, 40), (255, 255, 255), -1)
    # cv2.rectangle(image, (0, 40), (370, 370), (255, 255, 255), -1)

    # Noted that cv2.putText can only display English characters and not Chinese characters
    # We should convert it into chinese encoder
    image = cv2ImgAddText(image, "得分", 20, 10, (0, 153, 0), 30)
    # cv2.putText(image, str("Score"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
    height, width, _ = image.shape

    if angle_user[0] < (angle_target[0] - 15):
        # print("Extend the right arm at elbow")
        stage = stage + 1
        image = cv2ImgAddText(image, "在肘部伸展左臂", 10, 40, (0, 153, 0), 35)
        # cv2.putText(image, str("Extend the right arm at elbow"), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    # [0, 153, 0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[0][0] * width), int(angle_point[0][1] * height)), 30, (0, 0, 255), 5)

    if angle_user[0] > (angle_target[0] + 15):
        # print("Fold the right arm at elbow")
        stage = stage + 1
        image = cv2ImgAddText(image, "将左臂折叠在肘部", 10, 40, (0, 153, 0), 35)
        # cv2.putText(image, str("Fold the right arm at elbow"), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    # [0, 153, 0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[0][0] * width), int(angle_point[0][1] * height)), 30, (0, 0, 255), 5)

    if angle_user[1] < (angle_target[1] - 15):
        # print("Extend the left arm at elbow")
        stage = stage + 1
        image = cv2ImgAddText(image, "在肘部伸展右臂", 10, 75, (0, 153, 0), 35)
        # cv2.putText(image, str("Extend the left arm at elbow"), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    # [0, 153, 0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[1][0] * width), int(angle_point[1][1] * height)), 30, (0, 0, 255), 5)

    if angle_user[1] > (angle_target[1] + 15):
        # print("Fold the left arm at elbow")
        stage = stage + 1
        image = cv2ImgAddText(image, "将右臂折叠在肘部", 10, 75, (0, 153, 0), 35)
        # cv2.putText(image, str("Fold the left arm at elbow"), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    # [0, 153, 0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[1][0] * width), int(angle_point[1][1] * height)), 30, (0, 0, 255), 5)

    if angle_user[2] < (angle_target[2] - 15):
        # print("Lift your right arm")
        stage = stage + 1
        image = cv2ImgAddText(image, "抬起左臂", 10, 110, (0, 153, 0), 35)
        # cv2.putText(image, str("Lift your right arm"), (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2,
                    # cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[2][0] * width), int(angle_point[2][1] * height)), 30, (0, 0, 255), 5)

    if angle_user[2] > (angle_target[2] + 15):
        # print("Put your arm down a little")
        stage = stage + 1
        image = cv2ImgAddText(image, "把你的左胳膊放下一点", 10, 110, (0, 153, 0), 35)
        # cv2.putText(image, str("Put your arm down a little"), (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    # [0, 153, 0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[2][0] * width), int(angle_point[2][1] * height)), 30, (0, 0, 255), 5)

    if angle_user[3] < (angle_target[3] - 15):
        # print("Lift your left arm")
        stage = stage + 1
        image = cv2ImgAddText(image, "抬起右臂", 10, 145, (0, 153, 0), 35)
        # cv2.putText(image, str("Lift your left arm"), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2,
                    # cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[3][0] * width), int(angle_point[3][1] * height)), 30, (0, 0, 255), 5)

    if angle_user[3] > (angle_target[3] + 15):
        # print("Put your arm down a little")
        stage = stage + 1
        image = cv2ImgAddText(image, "把你的右胳膊放下一点", 10, 145, (0, 153, 0), 35)
        # cv2.putText(image, str("Put your arm down a little"), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    # [0, 153, 0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[3][0] * width), int(angle_point[3][1] * height)), 30, (0, 0, 255), 5)

    if angle_user[4] < (angle_target[4] - 15):
        # print("Extend the angle at right hip")
        stage = stage + 1
        image = cv2ImgAddText(image, "伸展左臀部的角度", 10, 180, (0, 153, 0), 35)
        # cv2.putText(image, str("Extend the angle at right hip"), (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    # [0, 153, 0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[4][0] * width), int(angle_point[4][1] * height)), 30, (0, 0, 255), 5)

    if angle_user[4] > (angle_target[4] + 15):
        # print("Reduce the angle at right hip")
        stage = stage + 1
        image = cv2ImgAddText(image, "减小左臀部的角度", 10, 180, (0, 153, 0), 35)
        # cv2.putText(image, str("Reduce the angle of at right hip"), (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    # [0, 153, 0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[4][0] * width), int(angle_point[4][1] * height)), 30, (0, 0, 255), 5)

    if angle_user[5] < (angle_target[5] - 15):
        # print("Extend the angle at left hip")
        stage = stage + 1
        image = cv2ImgAddText(image, "伸展右臀部的角度", 10, 215, (0, 153, 0), 35)
        # cv2.putText(image, str("Extend the angle at left hip"), (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    # [0, 153, 0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[5][0] * width), int(angle_point[5][1] * height)), 30, (0, 0, 255), 5)

    if angle_user[5] > (angle_target[5] + 15):
        # print("Reduce the angle at left hip")
        stage = stage + 1
        image = cv2ImgAddText(image, "缩小右臀部角度", 10, 215, (0, 153, 0), 35)
        # cv2.putText(image, str("Reduce the angle at left hip"), (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    # [0, 153, 0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[5][0] * width), int(angle_point[5][1] * height)), 30, (0, 0, 255), 5)

    if angle_user[6] < (angle_target[6] - 15):
        # print("Extend the angle of right knee")
        stage = stage + 1
        image = cv2ImgAddText(image, "伸展左膝角度", 10, 250, (0, 153, 0), 35)
        # cv2.putText(image, str("Extend the angle of right knee"), (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    # [0, 153, 0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[6][0] * width), int(angle_point[6][1] * height)), 30, (0, 0, 255), 5)

    if angle_user[6] > (angle_target[6] + 15):
        # print("Reduce the angle of right knee")
        stage = stage + 1
        image = cv2ImgAddText(image, "缩小左膝角度", 10, 250, (0, 153, 0), 35)
        # cv2.putText(image, str("Reduce the angle at right knee"), (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    # [0, 153, 0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[6][0] * width), int(angle_point[6][1] * height)), 30, (0, 0, 255), 5)

    if angle_user[7] < (angle_target[7] - 15):
        # print("Extend the angle at left knee")
        stage = stage + 1
        image = cv2ImgAddText(image, "伸展右膝角度", 10, 285, (0, 153, 0), 35)
        # cv2.putText(image, str("Extend the angle at left knee"), (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    # [0, 153, 0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[7][0] * width), int(angle_point[7][1] * height)), 30, (0, 0, 255), 5)

    if angle_user[7] > (angle_target[7] + 15):
        # print("Reduce the angle at left knee")
        stage = stage + 1
        image = cv2ImgAddText(image, "缩小右膝角度", 10, 285, (0, 153, 0), 35)
        # cv2.putText(image, str("Reduce th
        # e angle at left knee"), (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    # [0, 153, 0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[7][0] * width), int(angle_point[7][1] * height)), 30, (0, 0, 255), 5)
    if stage < 2:
        image = cv2ImgAddText(image, "很棒喔！", 170, 5, (0, 0, 225), 35)
        # cv2.putText(image, str("Great!"), (170, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2,
                    # cv2.LINE_AA)
    elif 2 < stage < 5:
        image = cv2ImgAddText(image, "继续加油！", 170, 5, (0, 0, 225), 35)
        # cv2.putText(image, str("Fighting!"), (170, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2, cv2.LINE_AA)

    elif stage > 5:
        image = cv2ImgAddText(image, "危险危险！", 170, 5, (225, 0, 0), 35)
        # cv2.putText(image, str("Dangerous!"), (170, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2,
                    # cv2.LINE_AA)
    return image


def Average(lst):
    return sum(lst) / len(lst)


def dif_compare(x, y):
    average = []
    for i, j in zip(range(len(list(x))), range(len(list(y)))):
        result = 1 - spatial.distance.cosine(list(x[i].values()), list(y[j].values()))
        average.append(result)
    score = math.sqrt(2 * (1 - round(Average(average), 2)))
    # print(Average(average))
    return score


def diff_compare_angle(x, y):
    new_x = []
    for i, j in zip(range(len(x)), range(len(y))):
        z = np.abs(x[i] - y[j]) / ((x[i] + y[j]) / 2)
        new_x.append(z)
        # print(new_x[i])
    return Average(new_x)


def compare_frame(camera_frame_image, target_frame_image):
    return 0


def detect_pose(cap, queue, H, W, S_w):
    # start the real-time pose detection and compare to the target
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        ptime = 0
        while cap.isOpened() and not isEnd:
            success, frame = cap.read()
            frame = cv2.flip(frame, 1)

            # time sleep here is very important because our pose is delayq
            time.sleep(0.05)
            if not success:
                print("Something error with your camera.")
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
            image = cv2.resize(image, (814, 610))

            try:
                image = cv2ImgAddText(image, "加油！", 10, 10, (0, 153, 0), 35)
                # cv2.putText(image, str("Excellent!"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2,
                            # cv2.LINE_AA)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3)
                                          )

                landmarks = results.pose_landmarks.landmark

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

                keypoints = []
                for point in landmarks:
                    keypoints.append({
                        'X': point.x,
                        'Y': point.y,
                        'Z': point.z,
                    })

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

                queue.put(keypoints)

            except:
                pass

            ctime = time.time()
            fps = 1 / (ctime - ptime)
            ptime = ctime
            fps_text = "FPS:" + str(int(fps)) + "ms"
            cv2.putText(image, str(fps_text), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, [254, 118, 136], 2,
                        cv2.LINE_AA)

            cv2.namedWindow('AI Exercise', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('AI Exercise', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # cv2.namedWindow('AI Exercise', 0)
            # cv2.resizeWindow('AI Exercise', 814, 644)
            cv2.resizeWindow('AI Exercise', int(0.45 * S_w), int(0.356 * S_w))
            # cv2.moveWindow('AI Exercise', 1104, 197)
            cv2.moveWindow('AI Exercise', W, int(H))
            cv2.imshow('AI Exercise', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                queue.put(None)
                cap.release()
                break

        queue.put(None)


def detect_video(cap, queue, H, W, S_w):
    # start the real-time pose detection and compare to the target
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and not isEnd:
            success, frame = cap.read()
           # frame = cv2.flip(frame, 1)

            if not success:
                print("End of your video")
                break

            # read frame image from real-time camera
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # set flag non-writeable and record the result arrays
            image.flags.writeable = False
            results = pose.process(image)

            # if we want to draw landmarks we need to make it writeable
            image.flags.writeable = True
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # target window screen height and width
            dim = (814, 644)
            image = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

            try:
                image = cv2ImgAddText(image, "跟着我做喔！", 10, 10, (0, 153, 0), 35)
                # cv2.putText(image, str("Follow me!"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2,
                            # cv2.LINE_AA)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3)
                                          )

                landmarks = results.pose_landmarks.landmark

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

                keypoints = []
                for point in landmarks:
                    keypoints.append({
                        'X': point.x,
                        'Y': point.y,
                        'Z': point.z,
                    })

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

                queue.put(keypoints)

            except:
                pass

            # display the video frame
            cv2.namedWindow('target_video', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('target_video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # cv2.namedWindow('target_video', 0)
            # cv2.resizeWindow('target_video', 814, 610)
            cv2.resizeWindow('target_video', int(0.45 * S_w), int(0.337 * S_w))
            # cv2.moveWindow('target_video', 290, 197)
            cv2.moveWindow('target_video', W, int(H))
            cv2.imshow('target_video', image)

            # show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)

            # self.cameraLabel.setPixmap(QPixmap.fromImage(showImage))
            if cv2.waitKey(10) & 0xFF == ord('q'):
                queue.put(None)
                cap.release()
                break

        queue.put(None)


def video_compare_pose(capqueue, tqueue, H, W):
    global isEnd
    while not isEnd:
        cap = capqueue.get()
        target = tqueue.get()

        # set a background
        background_file = "./pic/back.jpg"
        img = cv2.imread(background_file)
        cv2.line(img, (10, 40), (10, 400), (0, 238, 238))
        cv2.line(img, (10, 40), (40, 40), (0, 238, 238))
        cv2.line(img, (40, 40), (40, 400), (0, 238, 238))
        cv2.line(img, (10, 400), (40, 400), (0, 238, 238))
        # img = np.zeros((512, 512, 3), np.uint8)

        if cap is None or target is None:
            break

        global p_score
        p_score = dif_compare(cap, target)

        if p_score > 0.5:
            img = cv2ImgAddText(img, "小心！", 170, 5, (255, 0, 0), 35)
            # cv2.putText(img, str("Catch up!"), (170, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2,
                        # cv2.LINE_AA)
        elif 0.5 > p_score > 0.2:
            img = cv2ImgAddText(img, "很棒!", 170, 5, (255, 0, 0), 35)
            # cv2.putText(img, str("Good!"), (170, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2,
                        # cv2.LINE_AA)
        else:
            img = cv2ImgAddText(img, "完美！！！", 170, 5, (255, 0, 0), 35)
            # cv2.putText(img, str("Excellent!"), (170, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2,
                        # cv2.LINE_AA)

        # define per and bar
        color = (254, 118, 136)
        per = np.interp((1 - p_score), (0, 1.0), (0, 100))
        # per = (1 - avg_diff_norm) * 100
        bar = np.interp(1 - p_score, (0, 1.0), (400, 10))

        # Draw Bar
        cv2.putText(img, str("~p-score"), (150, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, [254, 130, 145], 2, cv2.LINE_AA)
        cv2.rectangle(img, (10, 400), (40, 400), color, cv2.FILLED)
        cv2.rectangle(img, (10, int(bar)), (40, 400), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)}', (60, 120), cv2.FONT_HERSHEY_PLAIN, 4,
                    color, 3)

        # Display the image in a window
        img = cv2.resize(img, (513, 512))
        cv2.namedWindow('Score', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Score', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # cv2.namedWindow('Score', 0)
        cv2.resizeWindow('Score', int(H), int(H))
        # cv2.moveWindow('Score', 24, 291)
        cv2.moveWindow('Score', int(W + ((W/9*11) - H)/2), 0)
        cv2.imshow('Score', img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            isEnd = True
            break



def voice_alert():
    global isEnd
    engine = pyttsx3.init()
    time.sleep(2)
    num = 2
    engine.say("3,   2,    1,   健身开始！")
    while not isEnd:
        # if video end, alert will end ahead of 3 second
        time.sleep(1)
        num += 1
        if clip_time - num <= 10:
            engine.say("视频即将播放结束")
            engine.runAndWait()
            num += 7
        if num >= clip_time - 3:
            isEnd = True
            break
        # alert
        if p_score < 0.5:
            engine.say("加油，请继续保持喔")
            engine.runAndWait()
            time.sleep(8)
            num += 8
        if p_score >= 0.6 and p_score <= 0.7:
            engine.say("危险，请注意调整动作")
            engine.runAndWait()
            time.sleep(3)
            num += 3
        if p_score > 0.7:
            engine.say("注意安全，容易受伤！")
            engine.runAndWait()
            num += 1.5
        if p_score <= 0.2 or isEnd:
            break
    engine.stop()


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)


