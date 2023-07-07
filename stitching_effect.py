import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread("./input/test.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"Original: width-{img.shape[0]} height-{img.shape[1]}")


def Extended_Image(img1, img2, model=1):
    if model == 1:
        if img1.shape[1] < img2.shape[1]:
            temp = img1.copy()
            img1 = img2.copy()
            img2 = temp

        add_height = img1.shape[0] - img2.shape[0]
        add_img = np.zeros((add_height, img2.shape[1], 3), np.uint8) + 255
        result = np.vstack((img2, add_img))

    elif model == 2:
        if img1.shape[0] < img2.shape[0]:
            temp = img1.copy()
            img1 = img2.copy()
            img2 = temp
        add_width = img1.shape[1] - img2.shape[1]
        add_img = np.zeros((img2.shape[0], add_width, 3), np.uint8) + 255
        result = np.hstack((img2, add_img))
    return result


# 下采样
def downsample(img, scale=2):
    h, w = img.shape[0:2]
    new_h = int(h / scale)
    new_w = int(w / scale)
    result = cv2.resize(img, (new_h, new_w), interpolation=cv2.INTER_LINEAR)
    return result


def init_guass(img, kernel=(3, 3)):
    return cv2.GaussianBlur(img, kernel, 1.5)


floors = int(float(input("Please enter the floors of the pyramid：")))

if floors == 1:
    print(f"当前输出的金字塔的层数为：{floors}")
    result = init_guass(img, kernel=(5, 5))

elif floors < 1:
    print("floors只能为大于0的整数")

elif floors > 1:
    print(f"Total Gaussian pyramid {floors} levels")
    img_1 = cv2.pyrDown(img)
    result = init_guass(img_1, kernel=(5, 5))
    img_1 = result.copy()
    for i in range(floors - 1):
        img_2 = downsample(img_1)
        img_2 = init_guass(img_2, kernel=(5, 5))
        temp_img = Extended_Image(img1=result, img2=img_2, model=2)
        result = np.vstack((result, temp_img))
        img_1 = img_2.copy()

    temp_img = Extended_Image(img1=img, img2=result, model=1)
    result = np.hstack((img, temp_img))

plt.figure(dpi=150)
plt.title(f"Gaussian pyramid\n(floors:{floors})")
plt.axis("off")
plt.imshow(result)
plt.savefig("./output/stitching_effect.png")
plt.show()