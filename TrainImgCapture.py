#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
# 作者：肖银皓
# 创建：2021-05-07
# 用意：利用TensorflowFaceDetector手动抓取脸部照片作为笑脸训练数据
# 注意事项：训练数据默认存储在项目根目录下的training_data文件夹中，该文件夹下面的子文件夹smile为带笑脸的训练数据，not_smile子文件夹为不带笑脸训练数据
"""
import collections

import numpy as np
import cv2
import copy
import FaceDetection.TensoflowFaceDetector as fd
from FaceDetection.utils import label_map_util
from FaceDetection.utils import visualization_utils_color as vis_util

# 人脸识别pre-trained model的路径
PATH_TO_CKPT = './FaceDetection/model/frozen_inference_graph_face.pb'

# 人脸框的标签路径
PATH_TO_LABELS = './FaceDetection/protos/face_label_map.pbtxt'

NUM_CLASSES = 2

# 训练数据存放地址
PATH_TO_SMILE_FACES = "./training_data/smile/"
PATH_TO_NOT_SMILE_FACES = "./training_data/not_smile/"

# 初始化图像和人脸坐标
image = None
boxes = None
image_cp = None
smile_index = 1800
not_smile_index = 1800

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def CaptureSmile(val):
    """
    这个函数为响应笑脸抓拍
    :param val: 无用
    :return: 无
    """
    # check if the value of the slider
    print("抓拍笑脸！")
    global smile_index
    if image is not None and boxes is not None:
        ymin, xmin, ymax, xmax = boxes[0][0]
        [h, w] = image.shape[:2]
        (x1, x2, y1, y2) = (int(xmin * w), int(xmax * w),
                                      int(ymin * h), int(ymax * h))
        cropped = image_cp[y1-1:y2+1, x1-1:x2+1]
        cv2.imwrite(PATH_TO_SMILE_FACES + str(smile_index) + ".jpg", cropped)
        smile_index += 1

def CaptureNotSmile(val):
    """
    这个函数为响应非笑脸抓拍
    :param val: 无用
    :return: 无
    """
    # check if the value of the slider
    print("抓拍非笑脸！")
    global not_smile_index
    if image is not None and boxes is not None:
        ymin, xmin, ymax, xmax = boxes[0][0]
        [h, w] = image.shape[:2]
        (x1, x2, y1, y2) = (int(xmin * w), int(xmax * w),
                            int(ymin * h), int(ymax * h)) # 标准化坐标，使得坐标跟随图片尺寸变化
        cropped = image_cp[y1:y2, x1:x2] # 裁剪
        cv2.imwrite(PATH_TO_NOT_SMILE_FACES + str(not_smile_index) + ".jpg", cropped)
        not_smile_index += 1

if __name__ == "__main__":

    camID = 0 #使用第一个摄像头
    tDetector = fd.TensoflowFaceDetector(PATH_TO_CKPT)

    SmileButton = [20, 60, 50, 250] #(y1,y2,x1,x2)
    NotSmileButton = [80, 60, 50, 250]
    cv2.namedWindow('Control')
    cv2.createTrackbar("Capture Smile", 'Control', 0, 1, CaptureSmile) #用Trackbar代替button因为Trackbar不需要QT支持
    cv2.createTrackbar("Capture Not Smile", "Control", 0, 1, CaptureNotSmile)

    cap = cv2.VideoCapture(camID)
    while True:
        ret, image = cap.read()
        if ret == 0:
            break

        [h, w] = image.shape[:2]
        image = cv2.flip(image, 1)
        image_cp = copy.copy(image)

        (boxes, scores, classes, num_detections) = tDetector.run(image)

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4)

        cv2.imshow("tensorflow based (%d, %d)" % (w, h), image)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break

    cap.release()
