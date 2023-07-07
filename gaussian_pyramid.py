import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread("./input/test.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"Original: width-{img.shape[0]} height-{img.shape[1]}")


# 下采样
def downsample(img, scale=0.5):
    """
    下采样
    :param img: 输入图像
    :param scale: 缩放比例，默认为0.5
    :return: 下采样后的图像
    """
    h, w = img.shape[0:2]
    new_h = int(h * scale)
    new_w = int(w * scale)
    # 采用双线性的方法
    result = cv2.resize(img, (new_h, new_w), interpolation=cv2.INTER_CUBIC)
    return result


# 获得子图的排布
def factor(num):
    """
    根据获得子图排布
    :param num: 金字塔层数
    :return: 子图排布
    """
    list1 = []
    for i in range(2, 10):
        temp = int(num / i)
        if (temp >= 1) & (temp * i == num):
            list1.append(i)
            list1.append(temp)
            break
    # list1.append(1) if len(list1) == 1 else None
    return list1


def gaussian_blur(img, K_size=3, sigma=1.3):
    """
    高斯滤波器
    :param img: 输入图像
    :param K_size: 核函数大小
    :param sigma: σ
    :return: 滤波后的图像
    """
    H, W, C = img.shape
    # Zero padding
    pad = K_size // 2
    # 黑边填充后会出现大黑边
    out = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
    # 用图像边缘像素进行填充
    # out = cv2.copyMakeBorder(img, pad, pad, pad, pad, borderType=cv2.BORDER_REPLICATE)
    # 创建和函数
    K = np.zeros((K_size, K_size), dtype=float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * sigma * sigma)
    K /= K.sum()
    tmp = out.copy()
    # 滤波
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])
    # 进行图像截断
    out = np.clip(out, 0, 255)
    # 转化格式：unit8
    out = out[pad: pad + H, pad: pad + W].astype("uint8")
    return out


# 交换
def swap(a, b): return (b, a) if a > b else (a, b)


# 生成图像高斯金字塔
def pyramid(imag, floors, guass_kernel):
    """
    生成图像高斯金字塔
    :param imag: 输入图像
    :param floors: 输入高斯金字塔层数
    :param guass_kernel: 高斯滤波核大小
    :return: None
    """
    # 打印当前金字塔层数
    if floors == 1:
        print(f"Total Gaussian pyramid {floors} levels")
        plt.figure(dpi=150, figsize=(8, 8))
        # 得到第零层
        # gus0 = gaussian_blur(imag, K_size=guass_kernel, sigma=1.3)
        plt.title("Gaussian pyramid\n 0 level")
        plt.imshow(img)
    else:
        print(f"Total Gaussian pyramid {floors} levels")
        x, y = factor(floors)[0:]
        # 保持x>y
        x, y = swap(x, y)
        # 打印布局大小
        print(f"Sub-map layout: row-{x} column-{y}")
        # 绘制图像
        plt.figure(dpi=100, figsize=(14, 8))
        plt.subplot(x, y, 1)
        # 得到第零层
        # gus0 = gaussian_blur(imag, K_size=guass_kernel, sigma=1.3)
        plt.title("Gaussian pyramid\n 0 level")
        plt.imshow(img)
        for i in range(2, floors + 1):
            imag = downsample(imag)
            imag = gaussian_blur(imag, K_size=guass_kernel, sigma=1.3)
            plt.subplot(x, y, i)
            plt.title(f"Gaussian pyramid\n {i - 1} level")
            plt.imshow(imag)
        # 设置子图间的水平和垂直间距
        plt.subplots_adjust(wspace=0.5, hspace=0.35)
    plt.savefig("./output/Gaussian_pyramid.png")
    plt.show()
    print("Image created successfully")


if __name__ == '__main__':
    floors = int(input("Please enter the floors of the pyramid\n（The input number is between 1 and 8）："))
    if (floors < 1) | (floors > 8):
        print("The floors must be between 1 and 8 ")
        exit()
    pyramid(img, floors=floors, guass_kernel=5)
