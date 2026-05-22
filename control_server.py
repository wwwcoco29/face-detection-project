#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
# 作者：肖银皓
# 创建：2021-05-10
# 用意：这个程序为树莓派等待网络信号，并控制两个C4G0101延时模块，带动直流电机打开门锁
# 注意事项1：该程序基于佛山市微风科技有限公司研发的C4G0101延时模块而开发，并配置成1000模式（上升沿触接通，发延时A断开），用其他模块可能要修改代码
# 注意事项2：该程序中C4G0101已通过USB配置好延时方式，具体环境需要具体分析延时模式
# 注意事项3：该程序基于python2.7开发，因为树莓派大多默认python2.7
"""

import RPi.GPIO as GPIO
from SimpleHTTPServer import SimpleHTTPRequestHandler
import BaseHTTPServer
import time

# 将14 15 pin设置成GPIO输出
GPIO.setmode(GPIO.BCM)
GPIO.setup(14, GPIO.OUT)
GPIO.setup(15, GPIO.OUT)

def moveUp():
    """
    接通线圈，带动上拉电机
    :return: 无
    """
    GPIO.output(14, 1)
    time.sleep(1)
    GPIO.output(14, 0)
    time.sleep(6)

def moveLeft():
    """
    接通线圈，带动左拉电机
    :return: 无
    """
    GPIO.output(15, 1)
    time.sleep(1)
    GPIO.output(15, 0)

class MyHandler(SimpleHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        if "leftonly" in self.path:
            print("收到仅左拉指令，执行......")
            moveLeft()
        elif "uponly" in self.path:
            print("收到仅上拉指令，执行......")
            moveUp()
        elif "unlock" in self.path:
            print("收到开锁指令，执行......")
            moveUp()
            moveLeft()


if __name__ == "__main__":
    HandlerClass = MyHandler
    ServerClass = BaseHTTPServer.HTTPServer
    server_address = ('', 8080)
    httpd = ServerClass(server_address, HandlerClass)

    print("serving on port 8080")
    httpd.serve_forever()
