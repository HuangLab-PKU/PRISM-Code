import cv2
import numpy as np
import os
from tqdm import tqdm
import re


def relabel_mask(
    intensity, plot_column=['CD_1_blur', 'CD_2_blur'],
    mask=np.array, mode="replace", 
    ori_label=int, ch_label=int,
    xlim=[-0.8, 0.8], ylim=[-0.6, 0.5],
    num_per_layer=15, G_layer=None,
):
    intensity_tmp = intensity.copy()
    x, y = mask.shape[0], mask.shape[1]

    if mode == "replace":
        if G_layer is None: data = intensity[intensity['G_layer'] == (ori_label-1)//num_per_layer]
        else: data = intensity[intensity['G_layer'] == G_layer]

    elif mode == "discard":
        data = intensity[intensity["label"] == ori_label]
        data["label"] = [-1] * len(data)

    data["x"] = (ylim[1] - data[plot_column[1]]) * x / (ylim[1] - ylim[0])
    data["y"] = (data[plot_column[0]] - xlim[0]) * y / (xlim[1] - xlim[0])
    data['x'] = data['x'].astype(int)
    data['y'] = data['y'].astype(int)
    mask_values = mask[data['x'].values, data['y'].values]
    
    data.loc[mask_values, 'label'] = ch_label
    intensity_tmp.loc[data.index, "label"] = data["label"]
    intensity_tmp = intensity_tmp[intensity_tmp["label"] != -1]

    return intensity_tmp


def relabel(intensity_fra, mask_dir, mode="discard", num_per_layer=15, xrange=[-0.8, 0.8], yrange=[-0.6, 0.8]):
    re_label = [match.group(1) for filename in os.listdir(mask_dir) if (match := re.search(r"mask_(\d+)\.png$", filename))]
    if len(re_label) == 0: return intensity_fra

    intensity_fra_relabel = intensity_fra.copy()
    for label in tqdm(sorted(list(map(int, re_label))), desc=f"Relabeling, mode={mode}"):
        mask = cv2.imread(os.path.join(mask_dir, f"mask_{label}.png"), cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(bool)
        intensity_fra_relabel = relabel_mask(
            intensity_fra_relabel,
            mask=mask,
            ori_label=label,
            ch_label=label,
            mode=mode,
            num_per_layer=num_per_layer,
            xlim=xrange, ylim=yrange,
        )

    return intensity_fra_relabel


# # Mouse callback function
# def draw_mask(event, x, y, flags, param):
#     global ix, iy, drawing
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix, iy = x, y

#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing:
#             cv2.line(mask, (ix, iy), (x, y), (255, 255, 255), 10)
#             ix, iy = x, y

#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         cv2.line(mask, (ix, iy), (x, y), (255, 255, 255), 10)
#         ix, iy = x, y


# def draw_mask_mannual(image_path, mask_path):
#     global image, mask
#     image = cv2.imread(image_path)
#     mask = np.zeros_like(image[:, :, 0])  # Create a single channel mask
#     cv2.namedWindow('image')
#     cv2.setMouseCallback('image', draw_mask)

#     while True:
#         # Blend the mask with the image
#         colored_mask = cv2.merge([mask, mask, mask])
#         overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)  # Adjust transparency here
        
#         cv2.imshow('image', overlay)
#         k = cv2.waitKey(1) & 0xFF
#         if k == 27:  # Press 'ESC' to exit
#             break

#     cv2.destroyAllWindows()

#     cv2.imwrite(mask_path, mask)


# 全局变量
drawing = False  # 鼠标是否按下
ix, iy = -1, -1  # 初始坐标
image = None
mask = None
points = []  # 存储绘制区域的点

# 鼠标回调函数
def draw_mask(event, x, y, flags, param):
    global ix, iy, drawing, image, mask, points

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        points = [(x, y)]  # 初始化点列表

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(image, (ix, iy), (x, y), (0, 0, 255), 1)
            cv2.line(mask, (ix, iy), (x, y), 255, 1)
            ix, iy = x, y
            points.append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # 绘制首尾相连的线段
        cv2.line(image, (ix, iy), points[0], (0, 0, 255), 1)
        cv2.line(mask, (ix, iy), points[0], 255, 1)
        # 计算所有点的几何中心作为种子点
        mean_x = int(sum(x for x, y in points) / len(points))
        mean_y = int(sum(y for x, y in points) / len(points))
        # mask[mean_y, mean_x] = 255  # 种子点设为255
        # print(mean_y, mean_x)
        # 使用floodFill填充
        flood_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
        # flood_mask[1:-1, 1:-1] = 1  # 除边界外其他部分设为1
        # flood_mask = np.ones_like(mask, np.uint8)
        cv2.floodFill(mask, flood_mask, (mean_x,mean_y), 255)
        # print(points)

def draw_mask_manual(image_path, mask_path):
    global image, mask
    image = cv2.imread(image_path)
    mask = np.zeros_like(image[:, :, 0])  # 创建单通道掩码
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_mask)

    while True:
        colored_mask = cv2.merge([mask, mask, mask])
        overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)  # Adjust transparency here
        cv2.imshow('image', overlay)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # 按 'ESC' 键退出
            break

    cv2.destroyAllWindows()
    cv2.imwrite(mask_path, mask)


def main():
    os.chdir(r"E:\TMC\PRISM_point_typing\dataset\PRISM30_mousebrain")

    # Global variables to store the last point and the mask
    last_point = None
    mask = None

    # Define the mouse callback function
    def draw_mask(event, x, y, flags, param):
        global last_point, mask
        if event == cv2.EVENT_LBUTTONDOWN:
            last_point = (x, y)
            cv2.circle(mask, (x, y), 10, (255, 255, 255), -1)
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            cv2.line(mask, last_point, (x, y), (255, 255, 255), 20)
            last_point = (x, y)

    # Display the image
    for layer in range(2):
        window_name = f"layer{layer}, 's' to save, 'c' to continue and 'q' to quit"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, draw_mask)

        img = cv2.imread(f"./figures/layer{layer+1}.jpg")
        mask = np.zeros(img.shape, dtype=np.uint8)

        while True:
            result = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
            cv2.imshow(window_name, result)

            if cv2.waitKey(1) & 0xFF == ord("s"):
                cluster = int(input("cluster you change: "))
                cv2.imwrite(f"./masks/mask_{cluster}.png", mask)
                break
            elif cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "main":
    main()
