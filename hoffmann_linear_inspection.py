import cv2
import math
import numpy as np
from skimage import util
import matplotlib.pyplot as plt


def HoughLines(image, angles_pace: int, upper_threshold: int, lower_threshold: int) -> list:
    # TODO 5:映射到霍夫曼空间
    # pace = 2
    print(f"角度步长为：{angles_pace}\n阈值范围为：{lower_threshold}~{upper_threshold}")
    # 量化角度，转化为弧度制
    angles_list = [math.radians(i) for i in range(0, 181, angles_pace)]

    # 获取点与r
    dot_list = []
    hough_dict = dict()

    # y
    for i in range(len(image)):
        # x
        for j in range(len(image[0])):
            # 判断是否为轮廓点
            if image[i][j] != 0:
                # 获取所有点的集合
                dot_list.append([j, i])
                for k in range(len(angles_list)):
                    # 根据θ计算r
                    r = j * math.cos(angles_list[k]) + i * math.sin(angles_list[k])
                    # 根据r与θ统计次数
                    try:
                        hough_dict[(round(r, 0), angles_list[k])] += 1
                    except:
                        hough_dict[(round(r, 0), angles_list[k])] = 1

    print(f"获取到点的数量：{len(dot_list)}")
    # 根据r的次数进行排序
    temp = (sorted(hough_dict.items(), key=lambda item: item[1]))
    print(temp)
    # for i in dot_list:
    #     x = i[0]
    #     y = i[1]
    #     plt.scatter(x, y)
    # plt.show()

    # TODO 6:取局部最大值，设置阈值，过滤干扰直线
    # 根据阈值过滤直线
    result_list = []
    # threshold = 85
    for i in temp:
        if (i[1] >= lower_threshold) & (i[1] <= upper_threshold):
            result_list.append(i[0])

    print(f"在阈值{lower_threshold}~{upper_threshold}的点数目为：{len(result_list)}\n具体为：{result_list}")
    # print(f"在阈值{lower_threshold}~{upper_threshold}的点数目为：{len(result_list)}")

    # TODO 7:绘制直线，标定交点
    # print(dot_list)

    x_list = []

    for line in result_list:
        r = line[0]
        angle = line[1]
        # 判断是否为第一个满足直线的点
        flog = 0
        for dot in dot_list:
            x = dot[0]
            y = dot[1]
            if math.sin(angle) != 0:
                y_ = (r - x * math.cos(angle)) / math.sin(angle)
            else:
                y_ = 0
            # 根据所得y值是否相近来判断是否为直线上的点
            if math.isclose(y, y_, rel_tol=0.001):
                if flog == 0:
                    # 设置起点与终点坐标
                    minx = maxx = x
                    miny = maxy = y
                    flog = 1
                else:
                    # 更新坐标
                    if x < minx:
                        minx = x
                        miny = y
                    elif x > maxx:
                        # 进行距离的测算，如果距离过远，则不算做终点
                        if math.sqrt((x - maxx) ** 2 + (y - maxy) ** 2) > 250:
                            continue
                        else:
                            maxx = x
                            maxy = y

        if (minx != maxx) | (miny != maxy):
            x_list.append([tuple((minx, miny)), tuple((maxx, maxy))])
    print("已获取所有直线的起点与终点")
    print(x_list)
    print("-" * 100)
    return x_list


def draw_lines(img, dot_list: list, output_path: str):
    """
    绘制Hough变换结果
    :param img: 背景图像
    :param dot_list: 绘制直线的点集
    :param output_path: 保存的路径
    :return: 绘制结束的图像
    """
    for dot in dot_list:
        dot1 = dot[0]
        dot2 = dot[1]
        cv2.line(img, dot1, dot2, (0, 0, 255), thickness=2)
    cv2.imwrite(output_path, img)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    image_path = "./input/lane2.jpg"
    # 读取图像
    img = cv2.imread(image_path)
    guass_noise_3 = (util.random_noise(image=img, mode='gaussian', var=0.03) * 255).astype(np.uint8)
    guass_noise_7 = (util.random_noise(image=img, mode='gaussian', var=0.07) * 255).astype(np.uint8)
    # print("图像基本信息：")
    # print(f"高:{img.shape[0]},宽:{img.shape[1]},通道数:{img.shape[2]}")

    # TODO 1:图像转化为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_noise_3 = cv2.cvtColor(guass_noise_3, cv2.COLOR_BGR2GRAY)
    gray_noise_7 = cv2.cvtColor(guass_noise_7, cv2.COLOR_BGR2GRAY)

    # TODO 2:去除噪声(高斯滤波)
    guassblur_img = cv2.GaussianBlur(gray_img, (3, 3), 1.5)
    guassblur_noise_3 = cv2.GaussianBlur(gray_noise_3, (5, 5), 1.5)
    guassblur_noise_7 = cv2.GaussianBlur(gray_noise_7, (7, 7), 1.5)

    # TODO 3:边缘提取 Canny
    # 3x3的Sobel算子
    canny_img = cv2.Canny(guassblur_img, 100, 200)
    cv2.imwrite("./output/canny.png", canny_img)
    canny_noise_3 = cv2.Canny(guassblur_noise_3, 100, 200)
    canny_noise_7 = cv2.Canny(guassblur_noise_7, 100, 200)
    # plt.imshow(canny_noise_7, cmap="gray")
    # plt.show()

    # TODO 4:二值化处理
    ret1, binary1 = cv2.threshold(canny_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite("./output/binary.png", binary1)
    ret2, binary2 = cv2.threshold(canny_noise_3, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ret3, binary3 = cv2.threshold(canny_noise_7, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # print(len(binary), len(binary[0]))

    dot_list = HoughLines(image=binary1, angles_pace=1, lower_threshold=92, upper_threshold=200)
    dot_list_noise_3 = HoughLines(image=binary2, angles_pace=1, lower_threshold=100, upper_threshold=200)
    dot_list_noise_7 = HoughLines(image=binary3, angles_pace=1, lower_threshold=80, upper_threshold=200)

    lines1 = draw_lines(img, dot_list, "./output/hough_lines1.png")
    lines2 = draw_lines(guass_noise_3, dot_list_noise_3, "./output/hough_lines2.png")
    lines3 = draw_lines(guass_noise_7, dot_list_noise_7, "./output/hough_lines3.png")

    title_list = ["Original\nHoughLines", "Guass_Noise_3\nHoughLines", "Guass_Noise_7\nHoughLines"]
    result_list = [lines1, lines2, lines3]
    plt.figure(figsize=(10, 8), dpi=150)
    for i in range(1, 4):
        plt.subplot(1, 3, i)
        plt.title(title_list[(i - 1)])
        plt.imshow(result_list[i - 1])
    plt.savefig("./output/focused_comparison.png", bbox_inches="tight")
    plt.show()
