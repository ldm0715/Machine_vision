import cv2
import numpy as np

# 实例化人脸分类器
# xml来源于资源文件
face_cascade = cv2.CascadeClassifier('./input/haarcascade_frontalface_default.xml')
# 读取测试图片
img = cv2.imread('./input/faces.jpg', cv2.IMREAD_COLOR)
# 将原彩色图转换成灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 开始在灰度图上检测人脸，输出是人脸区域的外接矩形框
# - scaleFactor：表示每次图像尺寸减小的比例因子。该参数可以控制级联分类器检测窗口的大小，以适应不同尺寸的对象。通常设置为1.1或1.2。
# - minNeighbors：指定每个候选矩形框周围要保留的邻居数量。该参数可以过滤掉一些错误检测结果，提高检测精度。通常设置为3至6之间。
# - minSize：指定对象的最小尺寸。如果检测到的对象太小，则可能是假阳性结果，应该过滤掉。通常设置为(30, 30)或(50, 50)。
# - maxSize：指定对象的最大尺寸。如果检测到的对象太大，则可能是假阳性结果，应该过滤掉
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1)
# 遍历人脸检测结果
for (x, y, w, h) in faces:
    # 在原彩色图上画人脸矩形框
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
# 显示画好矩形框的图片
cv2.namedWindow('faces', cv2.WINDOW_AUTOSIZE)
cv2.imshow('faces', img)
# 等待退出键
cv2.waitKey(0)
# 销毁显示窗口
cv2.destroyAllWindows()
