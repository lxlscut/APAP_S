import numpy as np


def warp_global(img, image_info, H):
    """
    :param img: 源图
    :param image_info: 映射的最终画布的信息
    :param H: 全局单应性矩阵
    :return: 转换完的数据
    """
    result_image = np.zeros([image_info.height, image_info.width, 3], dtype=np.uint8)
    h_inv = np.linalg.pinv(H)
    for i in range(image_info.height):
        for j in range(image_info.width):
            target_point = transfer(h_inv, src_point=np.array([j - image_info.offset_x, i - image_info.offset_y, 1]))
            # todo 映射在原图内，原图的值赋值过来
            if 0 < target_point[0] < img.shape[1] and 0 < target_point[1] < img.shape[0]:
                result_image[i, j, :] = img[int(target_point[1]), int(target_point[0]), :]
    return result_image


def transfer(H, src_point):
    target_point = H.dot(src_point.T)
    target_point = target_point / target_point[2]
    return target_point


def warp_local_homography_point(image_info, local_h, src, point):
    height = point[1]
    width = point[0]

    result_image = np.zeros([image_info.height, image_info.width, 3], dtype=np.uint8)
    for i in range(image_info.height):
        for j in range(image_info.width):
            m = 0
            n = 0
            while i >= height[m]:
                m += 1
            while j > width[n]:
                n += 1
            current_h = np.linalg.inv(local_h[m-1, n-1, :])
            target = transfer(current_h, np.array([j - image_info.offset_x, i - image_info.offset_y, 1]))
            if 0 < target[0] < src.shape[1] and 0 < target[1] < src.shape[0]:
                result_image[i, j, :] = src[int(target[1]), int(target[0]), :]
    return result_image
