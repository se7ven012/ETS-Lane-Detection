#%%
# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""
import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

import cv2
from PIL import Image
from scipy import stats
import numpy as np
import base
import windowsBase
#import imageio

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/ets.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/ets_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 生成目标边框颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None): #判断图片是否存在
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            #assert断言语句的语法格式 model_image_size[0][1]指图像的w和h，且必须是32的整数倍
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
            #letterbox_image()定义在utils.py的第20行。输入参数（图像 ,(w=416,h=416)),输出一张使用填充来调整图像的纵横比不变的新图。
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape) #（416，416,3）
        image_data /= 255. #归一化
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        #批量添加一维 -> (1,416,416,3) 为了符合网络的输入格式 -> (batch, w, h, c)

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            #目的为了求boxes,scores,classes，具体计算方式定义在generate（）函数内。在yolo.py第60行
            feed_dict={#喂参数
                self.yolo_model.input: image_data, #图像数据
                self.input_image_shape: [image.size[1], image.size[0]], #图像尺寸
                K.learning_phase(): 0 #学习模式 0=测试模型 1=训练模式
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        # 绘制边框，自动设置边框宽度，绘制边框和类别文字，使用Pillow绘图库

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32')) #字体
        thickness = (image.size[0] + image.size[1]) // 300 #厚度

        for i, c in reversed(list(enumerate(out_classes))): 
            predicted_class = self.class_names[c] #类别
            box = out_boxes[i] #框

            score = out_scores[i] #置信度

            label = '{} {:.2f}'.format(predicted_class, score) #标签
            draw = ImageDraw.Draw(image) #图画
            label_size = draw.textsize(label, font) #文字

            top, left, bottom, right = box 
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom)) #打印边框坐标

            if top - label_size[1] >= 0: #标签文字
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness): #画框
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])    
            draw.rectangle( #文字背景
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font) #文案
            del draw

        end = timer()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()

# CV车道检测工具
def do_canny(frame):
	# 将每一帧转化为灰度图像，去除多余信息
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	# 高斯滤波器，去除噪声，平滑图像
	blur = cv2.GaussianBlur(gray,(5,5),0)
	# 边缘检测
	# minVal = 50
	# maxVal = 150
	canny = cv2.Canny(blur,50,150)

	return canny

def do_segment(frame):
	# 获取图像高度(注意CV的坐标系,正方形左上为0点，→和↓分别为x,y正方向)
	height = frame.shape[0]

	# 创建一个区域,指定点
	polygons = np.array([
		[(400,730), 
		 (1280,730),
		 (860,500),
		 (820,500)]
		])

	# 创建一个mask,形状与frame相同，全为0值
	mask = np.zeros_like(frame)

	# 对该mask进行填充，做一个掩码
	# 三角形区域为1
	# 其余为0
	cv2.fillPoly(mask,polygons,255) 

	# 将frame与mask做与，抠取需要区域
	segment = cv2.bitwise_and(frame,mask) 
	return segment

def calculate_lines(frame,lines):
    try:
        # 建立两个空列表，用于存储左右车道边界坐标
        left = []
        right = []
        # 循环遍历lines
        for line in lines:
            # 将线段信息从二维转化能到一维
            x1,y1,x2,y2 = line.reshape(4)

            # 将一个线性多项式拟合到x和y坐标上，并返回一个描述斜率和y轴截距的系数向量
            parameters = np.polyfit((x1,x2), (y1,y2), 1)
            slope = parameters[0] #斜率 
            y_intercept = parameters[1] #截距

            # 通过斜率大小，可以判断是左边界还是右边界
            # 很明显左边界slope<0(注意cv坐标系不同的)
            # 右边界slope>0
            if slope < 0:
                left.append((slope,y_intercept))
            else:
                right.append((slope,y_intercept))

        # 将所有左边界和右边界做平均，得到一条直线的斜率和截距
        left_avg = np.average(left,axis=0)
        right_avg = np.average(right,axis=0)
        # 将这个截距和斜率值转换为x1,y1,x2,y2
        left_line = calculate_coordinate(frame,parameters=left_avg)
        right_line = calculate_coordinate(frame, parameters=right_avg)

        return np.array([left_line,right_line])
    except:
        return None

def calculate_coordinate(frame,parameters):
	# 获取斜率与截距
    try:
        slope, y_intercept = parameters
        # 设置初始y坐标为自顶向下(框架底部)的高度
        # 将最终的y坐标设置为框架底部上方500
        y1 = frame.shape[0]
        y2 = int(y1-500)
        # 根据y1=kx1+b,y2=kx2+b求取x1,x2
        x1 = int((y1-y_intercept)/slope)
        x2 = int((y2-y_intercept)/slope)
        return np.array([x1,y1,x2,y2]) 
    except TypeError:
        slope, y_intercept = 0, 0
        return np.array([]) 
    
def visualize_lines(frame,lines):
    lines_visualize = np.zeros_like(frame)
    try:
        if lines is not None:
            for x1,y1,x2,y2 in lines:
                cv2.line(lines_visualize,(x1,y1),(x2,y2),(0,0,255),5)
        return lines_visualize
    except:
        return np.zeros_like(frame)

def detect_video(yolo, video_path, output_path=""):
    win = base.getWinFromTitle(windowsBase.getWins(), "Simulator")
    if not win:
        print("can't find the window", end='')
        return False

    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path))
        out = cv2.VideoWriter(output_path)

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    image_list = []
    while True:
        #获取图像
        window = windowsBase.getWinPic(win[0])
        if window is False:
            print("can't get image", end="")
            break
        #处理图像
        b, g, r = window.split()
        image = Image.merge("RGB",(r,g,b))

        #道路识别部分
        frame = cv2.cvtColor(np.array(window),cv2.COLOR_RGB2BGR)
        canny = do_canny(frame)
        segment = do_segment(canny)
        hough = cv2.HoughLinesP(segment, 1, np.pi/180, 100, minLineLength=100, maxLineGap=50) 
        lines = calculate_lines(frame, hough)
        lines_visualize = visualize_lines(frame, lines)

        #物体识别部分
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.50, color=(255, 0, 0), thickness=2)
        result =cv2.addWeighted(result,1,lines_visualize,1,0.1)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)

        # image_list.append(result)
        # if len(image_list)>300:
        #     imageio.mimsave('pic.gif', image_list, duration=0.1)
        #     image_list = []
        #     break
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()



# %%
