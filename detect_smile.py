#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
# 作者：肖银皓
# 创建：2021-05-08
# 用意：利用LeNet-5对笑脸进行预测，并且利用笑脸
# 注意事项1：训练模型默认存放于根目录下的model文件夹中smile_detect_model.h5
"""
import copy
import threading
import time
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import http.client
import FaceDetection.TensoflowFaceDetector as fd
from FaceDetection.utils import label_map_util
from FaceDetection.utils import visualization_utils_color as vis_util

# 人脸识别pre-trained model的路径
PATH_TO_FACE_MODEL = './FaceDetection/model/frozen_inference_graph_face.pb'
# 人脸框的标签路径
PATH_TO_LABELS = './FaceDetection/protos/face_label_map.pbtxt'
NUM_CLASSES = 2
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
# 笑脸识别模型路径
PATH_TO_SMILE_MODEL = "./model/smile_detect_model.h5"

# 所有人脸照片默认resize至 224 x 224 x 3
face_img_height = 224
face_img_width = 224
face_img_depth = 3

smile_model = load_model(PATH_TO_SMILE_MODEL) # 加载笑脸模型

# 树莓派IP地址和控制端口
RASP_IP = '192.168.0.103'
RASP_PORT = 8080

# 笑容维持的时间阈值，默认3秒
smile_last_thresh = 1

def detect_smile(face_img):
    """
    识别是否为笑脸，如果是则返回1，否则返回0
    :param face_img: 脸部截取照片
    :return: 是笑脸返回0，不是返回1
    """
    face_resized = cv2.resize(face_img, (face_img_width, face_img_height))
    face_resized = img_to_array(face_resized)
    face_resized = np.array(face_resized, dtype='float') / 255.0
    roi = np.expand_dims(face_resized, axis=0)
    smile_result = smile_model.predict(roi).argmax()
    return smile_result

def send_cmd(cmd='unlock'):
    """
    发送http指令到树莓派
    :param cmd:有三种指令，`leftonly`, `uponly` 和 `unlock`，分别控制左拉电机，上拉电机以及解锁
    :return:
    """
    conn = http.client.HTTPConnection(RASP_IP, RASP_PORT)
    response = None
    if cmd == 'leftonly':
        print('仅左拉电机启动')
        conn.request("GET", "/leftonly")
        response = conn.getresponse()
    if cmd == 'uponly':
        print('仅上拉电机启动')
        conn.request("GET", "/uponly")
        response = conn.getresponse()
    if cmd == 'unlock':
        print('解锁......')
        conn.request("GET", "/unlock")
        response = conn.getresponse()
    else:
        print('无效指令！')

    if response is None or response.status != 200:
        print("指令响应失败，请检查连接！")

    conn.close()

class unlockThread(threading.Thread):
    """
    发送unlock http指令的线程
    """
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name

    def run(self):
        try:
            send_cmd()
        except ConnectionRefusedError:
            print("树莓派控制服务器未开启！")

if __name__ == "__main__":

    camID = 0 #使用第一个摄像头
    tDetector = fd.TensoflowFaceDetector(PATH_TO_FACE_MODEL)
    smile_start_time = -1

    cap = cv2.VideoCapture(camID)
    while True:
        ret, image = cap.read()
        if ret == 0:
            break

        [h, w] = image.shape[:2]
        image = cv2.flip(image, 1)
        image_cp = copy.copy(image)

        (boxes, scores, classes, num_detections) = tDetector.run(image)
        ymin, xmin, ymax, xmax = boxes[0][0]

        (x1, x2, y1, y2) = (int(xmin * w), int(xmax * w),
                            int(ymin * h), int(ymax * h))
        cropped = image_cp[y1:y2, x1:x2]

        try:
            result = detect_smile(cropped)
            str_list = []
            if result > 0:
                display_str = "Not Smiling"
                smile_start_time = -1
            else:
                if smile_start_time < 0: #开始计时笑脸时间
                    smile_start_time = time.time()
                else:
                    end = time.time()
                    elapsed = end - smile_start_time
                    if elapsed > smile_last_thresh:
                        print("笑容超过%d秒" % smile_last_thresh)
                        unlock_thread = unlockThread(1, "Thread-1")
                        unlock_thread.start()
                        smile_start_time = -1 #设置回初始值
                display_str = "Smiling"
            str_list.append(display_str)
            vis_util.draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax, color='Violet',
                                                      display_str_list=str_list)
        except cv2.error:
            pass

        cv2.imshow("tensorflow based (%d, %d)" % (w, h), image)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break

    cap.release()
