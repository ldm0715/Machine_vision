<h1 align="center">机器视觉课程的作业代码</h1>

## 前言
本仓库是机器视觉课程的代码实践，不会仅仅调用Python库进行实现，而是从原理出发，逐步实现算法。

本仓库的代码内容包括：
- [x] 作业一：图像高斯金字塔
- [x] 作业二：霍夫曼直线检测
- [x] 作业三：LBP特征提取算法
- [x] 作业四：OpenCV的人脸识别实践

## 仓库内容
### 1.图像高斯金字塔
#### 实验目的
- 学会图像的读取与存入。
- 利用高斯滤波器对不同尺度的图像进行平滑，生成高斯金字塔。
#### 实践结果
1. `gaussian_pyramid.py`：高斯金字塔的生成,从下采样与高斯滤波器两个方面逐步实现。**可调节生成金字塔的层数**，利用`Matplotlib`集中展示

<div align="center">
 <img src="https://z4a.net/images/2023/07/07/Gaussian_pyramid.png" alt="Gaussian_pyramid" width="60%">
</div>


2. `stitch_effect.py`：在实现高斯金字塔的前提下，**实现图像的拼接效果**。
<div align="center">
 <img src="https://z4a.net/images/2023/07/07/stitching_effect.png" alt="stitching_effect" width="60%">
</div>


### 2.霍夫曼直线检测
#### 实验目的
- 掌握霍夫曼直线检测的基本原理
- 针对两幅公路图片`lan1.jpg`和`lan2.jpg`，分别在无噪声、噪声3%和噪声7%的情况下，利用霍夫曼直线检测算法检测两幅图片中的直线。

#### 实践结果
`hoofmann_line_detection.py`：霍夫曼直线检测的实现，开放**角度角度步、最小阈值、最大阈值**三个参数。
<div align="center">
 <img src="https://z4a.net/images/2023/07/07/focused_comparison.png" alt="HoughLine" width="60%">
</div>


### 3.LBP特征提取算法
#### 实验目的
- 掌握LBP特征提取算法的基本原理
- 利用LBP特征提取算法提取lan1.jpg和lan2.jpg两幅图片的LBP特征，生成直方图显示；比较两幅图片的LBP特征直方图的差异。
#### 实践结果
1. `traditional_lbp.py`：传统LBP特征提取算法的实现。
<div align="center">
 <img src="https://z4a.net/images/2023/07/07/traditional_LBP.png" alt="traditional_LBP" width="60%">
</div>


2. `circle_lbp.py`：圆形LBP特征提取算法的实现。
<div align="center">
 <img src="https://z4a.net/images/2023/07/07/circular_LBP.png" alt="circular_LBP" width="60%">
</div>



### 4.OpenCV的人脸识别实践
粗略完成，后续有时间再补充。
<div align="center">
 <img src="https://z4a.net/images/2023/07/07/faces.jpg" alt="faces" width="60%">
</div>

## 仓库结构

~~~Dir Tree
机器视觉
├─ Course_design
│    ├─ face.py
│    └─ input
├─ README.md
├─ circular_LBP.py
├─ gaussian_pyramid.py
├─ hoffmann_linear_inspection.py
├─ input
│    ├─ Texturelabs_Metal_283S.jpg
│    ├─ Texturelabs_Metal_291S.jpg
│    ├─ gcnanmu.png
│    ├─ lane1.jpg
│    ├─ lane2.jpg
│    └─ test.png
├─ output
│    ├─ Gaussian_pyramid.png
│    ├─ binary.png
│    ├─ canny.png
│    ├─ circular_LBP.png
│    ├─ faces.jpg
│    ├─ focused_comparison.png
│    ├─ hough_lines1.png
│    ├─ hough_lines2.png
│    ├─ hough_lines3.png
│    ├─ stitching_effect.png
│    ├─ LBP.png
│    └─ traditional_LBP.png
├─ stitching_effect.py
└─ traditional_LBP.py
~~~

其中
- `input` - 文件夹为实验所需的图片
- `output` - 文件夹为实验结果。
- `Course_design` - OpenCV的人脸识别实践