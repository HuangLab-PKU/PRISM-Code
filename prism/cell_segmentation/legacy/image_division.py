import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt



# Running
def merge_overlap(
    image, labels,
    x_pos, x_num, x_st, cut_x,
    y_pos, y_num, y_st, cut_y,
    centroids, overlap_size,
):
    half_overlap_size = overlap_size // 2
    x_left_border = x_st if x_pos == 0 else x_st + half_overlap_size
    x_right_border = (image.shape[1] if x_pos == x_num - 1 else x_st + cut_x - half_overlap_size)
    y_left_border = y_st if y_pos == 0 else y_st + half_overlap_size
    y_right_border = (image.shape[2] if y_pos == y_num - 1 else y_st + cut_y - half_overlap_size)

    image[:, x_left_border: x_right_border, y_left_border: y_right_border] = labels[:, x_left_border - x_st: x_right_border - x_st, y_left_border - y_st: y_right_border - y_st,]

    for z in range(image.shape[0]):
        for x in range(x_left_border, x_right_border):
            pre_label = image[z, x, y_left_border - 1]
            post_label = image[z, x, y_left_border]
            if pre_label != 0 and post_label != 0 and pre_label != post_label:
                image[image == post_label] = pre_label

        for y in range(y_left_border, y_right_border):
            pre_label = image[z, x_left_border - 1, y]
            post_label = image[z, x_left_border, y]
            if pre_label != 0 and post_label != 0 and pre_label != post_label:
                image[image == post_label] = pre_label

    included_centroids = []
    for point in centroids:
        z, x, y = point + np.array([0, 1, 1])

        # Check if the point is within the specified range
        if x_left_border <= x < x_right_border and y_left_border <= y < y_right_border:
            included_centroids.append(point)

    return image, included_centroids


def divide_and_reconstract(model, image, overlap_size=50, max_volume=150*512*512, dtype=np.uint16):
    full = np.empty_like(image, dtype=dtype)

    centroid = []
    size =  int(np.sqrt(max_volume / image.shape[0]))
    x_num = -(-(image.shape[1] - overlap_size) // (size - overlap_size))
    y_num = -(-(image.shape[2] - overlap_size) // (size - overlap_size))

    cut_x = image.shape[1] // x_num + overlap_size
    cut_y = image.shape[2] // y_num + overlap_size

    print(f"n_tile: {x_num * y_num};", f"\nx_slice_num: {x_num};", f"y_slice_num: {y_num};")
    print(f"block_x: {cut_x};", f"block_y: {cut_y};", f"overlap: {overlap_size};")

    pad_width = [(0, 0), (1, 0), (1, 0)]
    image = np.pad(image, pad_width, constant_values=0)
    full = np.pad(full, pad_width, constant_values=0)

    x_st = 1
    y_st = 1
    x_step = cut_x - overlap_size
    y_step = cut_y - overlap_size

    cell_num = 0
    with tqdm(total=x_num * y_num, desc='tile') as pbar:
        for x_pos in range(x_num):
            for y_pos in range(y_num):
                cut = image[:, x_st : x_st + cut_x, y_st : y_st + cut_y]
                cut_labeled, details = model.predict_instances(cut, show_tile_progress=False, predict_kwargs={'verbose':0})
                cell_num_tmp = np.max(cut_labeled)

                cut_labeled += cell_num
                cut_labeled[cut_labeled == cell_num] = 0
                cut_centroids = [_ + np.array([0, x_st - 1, y_st - 1]) for _ in details["points"]]

                
                full, label_centroid = merge_overlap(
                    full, cut_labeled,
                    x_pos, x_num, x_st, cut_x, 
                    y_pos, y_num, y_st, cut_y, 
                    centroids=cut_centroids, 
                    overlap_size=overlap_size, 
                )
                
                centroid.append(label_centroid)
                cell_num += cell_num_tmp
                if cell_num > np.iinfo(dtype).max:
                    print('Warning: gray scale out of bound, decrease overlap or use higher bitmap may help.')
                    
                y_st += y_step
                pbar.update(1)

            x_st += x_step
            y_st = 0

    # remove the protect round
    return full[:, 1:, 1:], np.concatenate(centroid, axis=0)


def result_visualization(raw_image, predict_image):
    plt.figure(figsize=(10, 7), facecolor=(0.3,0.3,0.3))
    z = raw_image.shape[0] // 2
    y = raw_image.shape[1] // 2
    
    plt.subplot(221)
    plt.imshow(raw_image[z], cmap="gray")
    plt.axis("off")
    plt.title("XY slice")
    # tmp_ax.imshow(raw_image[raw_image.shape[0] // 2], cmap="gray")
    plt.subplot(222)
    plt.imshow(raw_image[:, y], cmap="gray")
    plt.axis("off")
    plt.title("XZ slice")
    
    plt.subplot(223)
    plt.imshow(raw_image[z], cmap="gray")
    plt.imshow(predict_image[z], cmap=lbl_cmap, alpha=0.5)
    plt.axis("off")
    plt.title("XY slice")
    
    plt.subplot(224)
    plt.imshow(raw_image[:, y], cmap="gray")
    plt.imshow(predict_image[:, y], cmap=lbl_cmap, alpha=0.5)
    plt.axis("off")
    plt.title("XZ slice")

    plt.tight_layout()
    plt.show()


def centroid_calculate_by_gray(dapi_predict, desc):
    voxel_coordinates = {}

    # Iterate through the array and store voxel coordinates
    for z in tqdm(range(dapi_predict.shape[0]),desc=desc):
        for x in range(dapi_predict.shape[1]):
            for y in range(dapi_predict.shape[2]):
                value = dapi_predict[z, x, y]
                if value == 0:
                    continue
                if value not in voxel_coordinates:
                    voxel_coordinates[value] = []
                voxel_coordinates[value].append((z, x, y))

    # Calculate centroids
    centroids = {}
    for value, coordinates in voxel_coordinates.items():
        coordinates = np.array(coordinates)
        centroid = coordinates.mean(axis=0)
        centroids[value] = centroid

    print(f'{desc}: {len(centroids)} nucleus.')
    centroids_df = pd.DataFrame(centroids)
    centroids_df = centroids_df.T
    centroids_df = centroids_df.sort_index()
    centroids_df.columns = ['z_in_pix', 'x_in_pix','y_in_pix']
    return centroids_df