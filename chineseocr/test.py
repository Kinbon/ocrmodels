import os
import json
import time
import web
import numpy as np
from PIL import Image
from config import *
from apphelper.image import union_rbox,adjust_box_to_origin,base64_to_PIL
from application import trainTicket,idcard 
import re
import csv
category = 0 # 对于不同的类别选择不同的框目前共四个类别
def sort_key(s):
    # 排序关键字匹配
    # 匹配开头数字序号
    if s:
        try:
            c = re.findall('yuwei1/d+', s)[0]
        except:
            c = -1
        return int(c)
if yoloTextFlag =='keras' or AngleModelFlag=='tf' or ocrFlag=='keras':
    if GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
        import tensorflow as tf
        from keras import backend as K
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.3## GPU最大占用量
        config.gpu_options.allow_growth = True##GPU是否可动态增加
        K.set_session(tf.Session(config=config))
        K.get_session().run(tf.global_variables_initializer())
    
    else:
      ##CPU启动
      os.environ["CUDA_VISIBLE_DEVICES"] = ''

if yoloTextFlag=='opencv':
    scale,maxScale = IMGSIZE
    from text.opencv_dnn_detect import text_detect
elif yoloTextFlag=='darknet':
    scale,maxScale = IMGSIZE
    from text.darknet_detect import text_detect
elif yoloTextFlag=='keras':
    scale,maxScale = IMGSIZE[0],2048
    from text.keras_detect import  text_detect
else:
     print( "err,text engine in keras\opencv\darknet")
     
from text.opencv_dnn_detect import angle_detect

if ocr_redis:
    ##多任务并发识别
    from helper.redisbase import redisDataBase
    ocr = redisDataBase().put_values
else:   
    from crnn.keys import alphabetChinese,alphabetEnglish
    if ocrFlag=='keras':
        from crnn.network_keras import CRNN
        if chineseModel:
            alphabet = alphabetChinese
            if LSTMFLAG:
                ocrModel = ocrModelKerasLstm
            else:
                ocrModel = ocrModelKerasDense
        else:
            ocrModel = ocrModelKerasEng
            alphabet = alphabetEnglish
            LSTMFLAG = True
            
    elif ocrFlag=='torch':
        from crnn.network_torch import CRNN
        if chineseModel:
            alphabet = alphabetChinese
            if LSTMFLAG:
                ocrModel = ocrModelTorchLstm
            else:
                ocrModel = ocrModelTorchDense
                
        else:
            ocrModel = ocrModelTorchEng
            alphabet = alphabetEnglish
            LSTMFLAG = True
    elif ocrFlag=='opencv':
        from crnn.network_dnn import CRNN
        ocrModel = ocrModelOpencv
        alphabet = alphabetChinese
    else:
        print( "err,ocr engine in keras\opencv\darknet")
     
    nclass = len(alphabet)+1   
    if ocrFlag=='opencv':
        crnn = CRNN(alphabet=alphabet)
    else:
        crnn = CRNN( 32, 1, nclass, 256, leakyRelu=False,lstmFlag=LSTMFLAG,GPU=GPU,alphabet=alphabet)
    if os.path.exists(ocrModel):
        crnn.load_weights(ocrModel)
    else:
        print("download model or tranform model with tools!")
        
    ocr = crnn.predict_job
    
   
from main import TextOcrModel

model =  TextOcrModel(ocr,text_detect,angle_detect)
from apphelper.image import xy_rotate_box,box_rotate,solve
import cv2
import numpy as np
def plot_box(img,boxes):
    blue = (0, 0, 0) #18
    tmp = np.copy(img)
    for box in boxes:
         cv2.rectangle(tmp, (int(box[0]),int(box[1])), (int(box[2]), int(box[3])), blue, 1) #19
    
    return Image.fromarray(tmp) 

def plot_boxes(img,angle, result,color=(0,0,0)):
    tmp = np.array(img)
    c = color
    h,w = img.shape[:2]
    thick = int((h + w) / 300)
    i = 0
    if angle in [90,270]:
        imgW,imgH = img.shape[:2]
        
    else:
        imgH,imgW= img.shape[:2]

    for line in result:
        cx =line['cx']
        cy = line['cy']
        degree =line['degree']
        w  = line['w']
        h = line['h']

        x1,y1,x2,y2,x3,y3,x4,y4 = xy_rotate_box(cx, cy, w, h, degree/180*np.pi)
        
        x1,y1,x2,y2,x3,y3,x4,y4 = box_rotate([x1,y1,x2,y2,x3,y3,x4,y4],angle=(360-angle)%360,imgH=imgH,imgW=imgW)
        cx  =np.mean([x1,x2,x3,x4])
        cy  = np.mean([y1,y2,y3,y4])
        cv2.line(tmp,(int(x1),int(y1)),(int(x2),int(y2)),c,1)
        cv2.line(tmp,(int(x2),int(y2)),(int(x3),int(y3)),c,1)
        cv2.line(tmp,(int(x3),int(y3)),(int(x4),int(y4)),c,1)
        cv2.line(tmp,(int(x4),int(y4)),(int(x1),int(y1)),c,1)
        mess=str(i)
        cv2.putText(tmp, mess, (int(cx), int(cy)),0, 1e-3 * h, c, thick // 2)
        i+=1
    return Image.fromarray(tmp).convert('RGB')


def plot_rboxes(img,boxes,color=(0,0,0)):
    tmp = np.array(img)
    c = color
    h,w = img.shape[:2]
    thick = int((h + w) / 300)
    i = 0


    for box in boxes:

        x1,y1,x2,y2,x3,y3,x4,y4 = box
        
        
        cx  =np.mean([x1,x2,x3,x4])
        cy  = np.mean([y1,y2,y3,y4])
        cv2.line(tmp,(int(x1),int(y1)),(int(x2),int(y2)),c,1)
        cv2.line(tmp,(int(x2),int(y2)),(int(x3),int(y3)),c,1)
        cv2.line(tmp,(int(x3),int(y3)),(int(x4),int(y4)),c,1)
        cv2.line(tmp,(int(x4),int(y4)),(int(x1),int(y1)),c,1)
        mess=str(i)
        cv2.putText(tmp, mess, (int(cx), int(cy)),0, 1e-3 * h, c, thick // 2)
        i+=1
    return Image.fromarray(tmp).convert('RGB')
import time
from PIL import Image
import glob
ps = glob.glob('./yuwei2/*.JPG')
birth_data = []
with open('./k2_label.csv') as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    birth_header = next(csv_reader)  # 读取第一行每一列的标题
    for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
        birth_data.append(row)
# p = './test/idcard-demo.jpeg'
all_count = 0
all_correct = 0
for p in ps:
    each_correct = 0
    all_count += 68
    category = 1
    img = cv2.imread(p)

    h,w = img.shape[:2]
    timeTake = time.time()
    scale=608
    maxScale=2048

    result,angle= model.model(img,category,
                                        detectAngle=False,##是否进行文字方向检测
                                        scale=scale,
                                        maxScale=maxScale,
                                        MAX_HORIZONTAL_GAP=80,##字符之间的最大间隔，用于文本行的合并
                                        MIN_V_OVERLAPS=0.6,
                                        MIN_SIZE_SIM=0.6,
                                        TEXT_PROPOSALS_MIN_SCORE=0.1,
                                        TEXT_PROPOSALS_NMS_THRESH=0.7,
                                        TEXT_LINE_NMS_THRESH = 0.9,##文本行之间测iou值
                                        LINE_MIN_SCORE=0.1,                                             
                                        leftAdjustAlph=0,##对检测的文本行进行向左延伸
                                        rightAdjustAlph=0.1,##对检测的文本行进行向右延伸
                                        
                                    )
            
    timeTake = time.time()-timeTake

    print('It take:{}s'.format(timeTake))
    if category == 0:
        for i in range(17):
            if result[i - 34]['text'] == str(birth_data[(int(p.split('/')[-1].split('.JPG')[0]) - 1) * 2][i]):
                each_correct += 1
            else:
                result[i - 34]['img'].save('error_img/' + str(all_count - 34 + i) + '.JPG')
                with open('error_img/' + str(all_count - 34 + i) + '.txt', 'w') as f:
                    f.write(str(birth_data[(int(p.split('/')[-1].split('.JPG')[0]) - 1) * 2][i]))
                with open('error_img/' + str(all_count - 34 + i) + '_wrong.txt', 'w') as f1:
                    f1.write(result[i - 34]['text'])
        for i in range(17, 34):
            if result[i - 34]['text'] == str(birth_data[(int(p.split('/')[-1].split('.JPG')[0]) - 1) * 2 + 1][i - 17]):
                each_correct += 1
            else:
                result[i - 34]['img'].save('error_img/' + str(all_count - 34 + i) + '.JPG')
                with open('error_img/' + str(all_count - 34 + i) + '.txt', 'w') as f:
                    f.write(str(birth_data[(int(p.split('/')[-1].split('.JPG')[0]) - 1) * 2 + 1][i - 17]))
                with open('error_img/' + str(all_count - 34 + i) + '_wrong.txt', 'w') as f1:
                    f1.write(result[i - 34]['text'])
        print('The ' + p.split('/')[-1].split('.JPG')[0] + ' pic is ' + str(each_correct / 34) + ' accuracy')
        all_correct += each_correct
    elif category == 1:
        img_name = p.split('/')[-1].split('.JPG')[0]
        for row in birth_data:
            if row != []:
                if row[0] == img_name:
                    for i in range(68):
                        if result[i - 68]['text'] == str(row[i + 1]).split(' ')[0]:
                            each_correct += 1
                        else:
                            result[i - 68]['img'].save('error_img/' + str(all_count - 68 + 2184 + i) + '.JPG')
                            with open('error_img/' + str(all_count + 2184 - 68 + i) + '.txt', 'w') as f:
                                f.write(str(row[i + 1]))
                            with open('error_img/' + str(all_count + 2184 - 68 + i) + '_wrong.txt', 'w') as f1:
                                f1.write(result[i - 68]['text'])
        print('The ' + p.split('/')[-1].split('.JPG')[0] + ' pic is ' + str(each_correct / 68) + ' accuracy')
        all_correct += each_correct
print('All pic is ' + str(all_correct / all_count) + ' accuracy')