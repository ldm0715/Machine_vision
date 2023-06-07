import cv2
import numpy as np
import matplotlib.pyplot as plt


# 参考1：https://blog.csdn.net/hongbin_xu/article/details/79924961
# 参考2：https://blog.csdn.net/literacy_wang/article/details/106742890

def init_image(input_path):
    """
    初始化图像
    :param input_path: 图像路径
    :return: 处理后的图像
    """
    # 以单通道的形式读取图像
    image = cv2.imread(input_path, 0)
    # 给图像填充边缘像素
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, borderType=cv2.BORDER_REPLICATE)
    h, w = image.shape[0], image.shape[1]
    # 将灰度值转化为int32，方便后续计算
    image = image.astype(np.int32)
    image_name = input_path.split("/")[-1]
    print(f"图像名：{image_name}，高：{h}, 宽：{w}")
    return image


# 进制转化方法
# int(x, base)
# x - 字符 1010
# base - 字符对应的进制 2

def transform(num):
    if num > 0:
        return 1
    else:
        return 0


def LBP(image):
    """
    LBP算法
    :param image: 待处理图像
    :param h: 图像的
    :param w:
    :return:
    """
    h, w = image.shape[0], image.shape[1]
    # 处理后的图像
    temp = np.zeros((h - 1, w - 1))
    print("---图像处理中---")
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            calculation_list = list()
            calculation_list.append(image[i - 1][j - 1] - image[i][j])
            calculation_list.append(image[i][j - 1] - image[i][j])
            calculation_list.append(image[i + 1][j - 1] - image[i][j])

            calculation_list.append(image[i + 1][j] - image[i][j])
            calculation_list.append(image[i + 1][j + 1] - image[i][j])

            calculation_list.append(image[i][j + 1] - image[i][j])
            calculation_list.append(image[i - 1][j + 1] - image[i][j])
            calculation_list.append(image[i - 1][j] - image[i][j])
            # 将所得值转化为0，1
            for k in range(8):
                calculation_list[k] = transform(calculation_list[k])
            # 获得二进制码
            result_str = "".join([str(i) for i in calculation_list])
            # 转化为十进制并存入图像
            temp[i - 1][j - 1] = int(result_str, 2)
        # print(result_str)
    # 将像素值归一化
    temp = np.clip(temp, 0, 255).astype("uint8")
    # 转化通道，方便后续展示
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    print("---图像处理完成---")
    return temp


def show_result(image_list, title_list):
    """
    展示处理后的图像及其像素直方图
    :param image_list: 图像列表
    :param title_list: 标题列表
    :return: None
    """
    # 设定图的大小
    plt.figure(figsize=(8, 5), dpi=100)
    # 绘图
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title(title_list[i])
        # 将直方图放在第二列
        if i % 2 == 1:
            plt.xticks(range(0, 257, 16), range(0, 17))
            plt.hist(image_list[int(i / 2)].ravel(), 16)
            plt.xlabel("Dimension")
            plt.ylabel("Value")
        # 处理图放在第一列
        else:
            plt.axis("off")
            plt.imshow(image_list[int(i / 2)])
    # 紧凑排布
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    input_path1 = "input/lane1.jpg"
    image = init_image(input_path1)
    result1 = LBP(image)

    input_path2 = "input/lane2.jpg"
    image = init_image(input_path2)
    result2 = LBP(image)

    image_list = [result1, result2]
    title_list = ["lane1_LBP", "lane1_LBP_hist", "lane2_LBP", "lane2_LBP_hist"]
    show_result(image_list, title_list)
