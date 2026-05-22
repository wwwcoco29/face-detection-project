#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
# 作者：肖银皓
# 创建：2021-05-08
# 用意：利用LeNet-5对笑脸进行训练
# 注意事项1：训练数据默认存储在项目根目录下的training_data文件夹中，该文件夹下面的子文件夹smile为带笑脸的训练数据，not_smile子文件夹为不带笑脸训练数据
# 注意事项2：训练模型默认存放于根目录下的model文件夹中smile_detect_model.h5
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import Input
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from LeNet import LeNet
import numpy as np
import os.path
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # 只用CPU进行训练，如果要用GPU训练请注释起来

# 默认路径
PATH_TO_MODEL = "./model/smile_detect_model.h5"
PATH_TO_TRAIN_DATA = "./training_data/"

# 所有人脸照片默认resize至 224 x 224 x 3
face_img_height = 224
face_img_width = 224
face_img_depth = 3

data = []
labels = []

# 循环所有的照片，并将照片数据和标签格式化
for dirpath, dirnames, filenames in os.walk(PATH_TO_TRAIN_DATA):
    for filename in [f for f in filenames]:
        if '.jpg' not in filename and '.png' not in filename:
            continue
        img_path = dirpath + '/' + filename
        im = cv2.imread(img_path)
        im_resized = cv2.resize(im, (face_img_width, face_img_height))
        im_resized = img_to_array(im_resized)
        data.append(im_resized)

        # 打标签
        if 'not_smile' in img_path:
            label = [0, 1]
        else:
            label = [1, 0]
        labels.append(label)

# 将像素变为 [0,1] 之间的小数
data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)

loop = 1

while loop <= 100:
    print("开始第%d次训练" % loop)
    # 将训练数据拆分为90%作为训练数据，20%作为测试数据，打乱数据
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

    # 初始化模型
    print('正在构建DenseNet121......')
    input_tensor = Input(shape=(face_img_width, face_img_height, face_img_depth))
    model = LeNet.build(width=face_img_width, height=face_img_height, depth=face_img_depth, classes=2)
    model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy'])

    # 如果有已经存在的模型，加载继续训练
    if os.path.exists(PATH_TO_MODEL):
        print("载入已有模型：" + PATH_TO_MODEL)
        model = load_model(PATH_TO_MODEL)

    # 训练模型
    print('开始训练模型...')
    H = model.fit(trainX, trainY, validation_data=(testX, testY), class_weight=None, batch_size=128, epochs=5, verbose=1)

    # 评估模型
    print('[INFO] evaluating network...')
    predictions = model.predict(testX, batch_size=128)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1)))

    # 保存模型
    print('序列化模型并保存......')
    model.save(PATH_TO_MODEL)
    loop += 1