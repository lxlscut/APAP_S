import numpy as np


class Image_info:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.offset_x = 0
        self.offset_y = 0
        self.left_top = np.array([0, 0])
        self.right_top = np.array([0, 0])
        self.left_button = np.array([0, 0])
        self.right_button = np.array([0, 0])

    def get_final_size(self, src_img, dst_img, H):
        """
        :param src_img: 需要转换的图片
        :param dst_img: 不需要转换的图片
        :param H: 单应性矩阵
        :return: 图片的size,以及x，y方向的偏移量
        """

        offset_x = 0
        offset_y = 0
        # todo 第一步：计算转换后的四个顶点

        # todo calculate top left corner coordinate
        corner = []
        v2 = np.array([0, 0, 1], dtype=np.float64)
        v1 = H.dot(v2)
        corner.append([v1[0] / v1[2], v1[1] / v1[2]])
        self.left_top = np.array([v1[0] / v1[2], v1[1] / v1[2]])
        # todo calculate the coordinate of left bottom
        v2 = np.array([0, src_img.shape[0], 1], dtype=np.float64)
        v1 = H.dot(v2)
        corner.append([v1[0] / v1[2], v1[1] / v1[2]])
        self.left_button = np.array([v1[0] / v1[2], v1[1] / v1[2]])
        #  todo calculate the coordinate of right top
        v2 = np.array([src_img.shape[1], 0, 1], dtype=np.float64)
        v1 = H.dot(v2)
        corner.append([v1[0] / v1[2], v1[1] / v1[2]])
        self.right_top = np.array([v1[0] / v1[2], v1[1] / v1[2]])
        #  todo calculate the coordinate of right top
        v2 = np.array([src_img.shape[1], src_img.shape[0], 1], dtype=np.float64)
        v1 = H.dot(v2)
        corner.append([v1[0] / v1[2], v1[1] / v1[2]])
        self.right_button = np.array([v1[0] / v1[2], v1[1] / v1[2]])
        corner = np.array(corner).astype(np.int)

        # todo x,y的最大与最小值
        max_x = max(np.max(corner[:, 0]), dst_img.shape[1])
        max_y = max(np.max(corner[:, 1]), dst_img.shape[0])

        min_x = min(np.min(corner[:, 0]), 0)
        min_y = min(np.min(corner[:, 1]), 0)

        w = max_x - min_x
        h = max_y - min_y
        self.width = w
        self.height = h
        if min_x < 0:
            offset_x = 0 - min_x

        if min_y < 0:
            offset_y = 0 - min_y
        self.offset_x = offset_x
        self.offset_y = offset_y
