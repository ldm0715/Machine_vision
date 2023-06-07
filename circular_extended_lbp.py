import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


# 参考资料：https://blog.csdn.net/literacy_wang/article/details/106742890

def init_image(input_path, R):
    """
    初始化图像
    :param input_path: 图像路径
    :param R:范围圆的半径
    :return:处理后的图像
    """
    # 单通道读取
    image = cv2.imread(input_path, 0)
    h, w = image.shape[0], image.shape[1]
    # 四周添加R的边缘像素
    image = cv2.copyMakeBorder(image, R, R, R, R, borderType=cv2.BORDER_REPLICATE)
    # 转化数据类型为int32，方便后续计算
    image = image.astype(np.int32)
    image_name = input_path.split("/")[-1]
    print(f"图像名：{image_name}，高：{h}, 宽：{w}")
    return image


def transform(num):
    # print(num)
    if num > 0:
        return 1
    else:
        return 0


# 双线性插值
def bilinear_interpolation(xp, yp, image):
    """
    双线性插值
    :param xp: x坐标
    :param yp: y坐标
    :param image: 输入的图像
    :return: 计算后所得（x，y）的灰度值
    """
    h, w = image.shape[0], image.shape[1]
    # 向下取整
    x1, y1 = int(xp), int(yp)
    # 获得四点的坐标
    if x1 > w - 1:
        x2 = x1 - 1
    else:
        x2 = x1 + 1
    if y1 > h - 1:
        y2 = y1 - 1
    else:
        y2 = y1 + 1
    f_x1y1 = image[y1, x1]
    f_x1y2 = image[y2, x1]
    f_x2y1 = image[y1, x2]
    f_x2y2 = image[y2, x2]
    # 计算插值
    pixel_value = f_x1y1 * (x2 - xp) * (y2 - yp) + f_x2y1 * (xp - x1) * (y2 - yp) + \
                  f_x1y2 * (x2 - xp) * (yp - y1) + f_x2y2 * (xp - x1) * (yp - y1)
    return pixel_value


# 判断是否为整数
def init_integer(num):
    if num.is_integer():
        return int(num)
    else:
        return num


def get_min(str, p):
    """
    得到最小值编码
    :param str:二进制字符串
    :param p:获取点的个数
    :return:十进制最小的值
    """
    min = int(str, 2)
    for i in range(len(str) - 1):
        temp = str[1:p]
        temp = temp + str[0]
        str = temp
        if int(str, 2) < min:
            min = int(str, 2)
    return min


def circular_extended_lbp(image_path, r, p):
    # 初始化图像
    image = init_image(image_path, r)
    # 获得图像的宽高
    h, w = image.shape[0], image.shape[1]
    print(f"开始LBP处理，设定参数：半径：{r} 点个数：{p}")
    start_time = time.time()
    # 处理后的图像
    temp = np.zeros((h - r, w - r))
    for i in range(r, h - r):
        for j in range(r, w - r):
            calculation_list = list()
            for k in range(p):
                # 转化为弧度制
                radians = np.radians(2 * np.pi * k / float(p))
                # 为了对应图像坐标系进行减一操作
                xp = j + r * np.cos(radians) - 1
                yp = i - r * np.sin(radians) - 1
                xp = init_integer(xp)
                yp = init_integer(yp)
                # print(xp, yp)
                # 计算与圆心点的灰度值的差值
                try:
                    calculation_list.append(image[xp][yp] - image[i][j])
                except:
                    calculation_list.append(bilinear_interpolation(xp, yp, image) - image[i][j])
                # 获得二进制字符串
                calculation_list[k] = transform(calculation_list[k])
            # print(calculation_list)
            result_str = "".join([str(i) for i in calculation_list])
            # 得到二进制中最小的十进制编码数
            temp[i - r][j - r] = get_min(result_str, p)
    temp = np.clip(temp, 0, 255).astype("uint8")
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    end_time = time.time()
    all_time = end_time - start_time
    print(f"LBP处理完毕，耗时：{int(all_time // 60)}m{int(all_time % 60)}s")
    print("-" * 100)
    return temp


def show_result(image_list, title_list):
    plt.figure(figsize=(8, 5))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title(title_list[i])
        if i % 2 == 1:
            plt.xticks(range(0, 257, 16), range(0, 17))
            plt.hist(image_list[int(i / 2)].ravel(), 16)
            plt.xlabel("Dimension")
            plt.ylabel("Value")
        else:
            plt.axis("off")
            plt.imshow(image_list[int(i / 2)])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 圆的半径
    R = 2
    # 取多少个点
    P = 6
    input_path = "./input/lane1.jpg"
    result_image1 = circular_extended_lbp(input_path, R, P)

    input_path2 = "./input/lane2.jpg"
    result_image2 = circular_extended_lbp(input_path, R, P)

    image_list = [result_image1, result_image2]
    title_list = ["lane1_circular_LBP", "lane1_circular_LBP_hist", "lane2_circular_LBP", "lane2_circular_LBP_hist"]
    show_result(image_list, title_list)
