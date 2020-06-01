import math
import numpy as np
import copy

# todo 使用RANSAC算法来筛选特征点
from UTIL.calculate_homography import get_final_homography
from UTIL.glabal_homography import GloablHomography


class RANSAC:
    def __init__(self, p=0.995, epsilon=0.5, threshold=10, src_point=None, dst_point=None):
        self.p = p
        self.epsilon = epsilon
        self.threshold = threshold
        self.src_point = src_point
        self.dst_point = dst_point
        self.N = self.calculate_N()

    # 计算一共需要进行多少轮的计算
    def calculate_N(self):
        return int(math.log(1 - self.p) / math.log(1 - math.pow((1 - self.epsilon), 6) + 1e-8))

    # todo 进行ransac计算
    def ransac(self):
        src_point = copy.deepcopy(self.src_point)
        dst_point = copy.deepcopy(self.dst_point)

        src_for_err = copy.deepcopy(src_point)
        dst_point_err = copy.deepcopy(dst_point)

        # todo 最终的选出的特征点的个数
        final_num = 0
        gh = GloablHomography()
        for i in range(min(self.N, 2000)):
            # todo 选取其中的四个点
            chose_point = np.arange(len(src_point))
            np.random.shuffle(chose_point)

            new_src_array = src_point[chose_point[:6], :]
            new_dst_array = dst_point[chose_point[:6], :]

            # # todo 判断其是否共线
            # xy1 = new_src_array[0, :] - new_src_array[1, :]
            # k1 = xy1[1] / xy1[0]
            #
            # xy2 = new_src_array[2, :] - new_src_array[3, :]
            # k2 = xy2[1] / xy2[0]
            # print(k1,k2)
            # if abs(k1 - k2) < 2:
            #     continue

            cal_src = copy.deepcopy(new_src_array)
            cal_dst = copy.deepcopy(new_dst_array)
            h = gh.get_global_homo(cal_src, cal_dst)

            target = np.zeros_like(src_for_err)
            error = np.zeros_like(src_for_err)
            for j in range(src_point.shape[0]):
                target[j] = self.my_transform(src_for_err[j], h)
                error[j] = target[j] - dst_point_err[j]
            err = np.sqrt(error[:, 0] ** 2 + error[:, 1] ** 2)
            point_ok = np.where(err < self.threshold)
            if point_ok[0].shape[0] > final_num:
                final_num = point_ok[0].shape[0]
                # todo 更新循环次数,如果全都是外点的话（包括自身的4个点），
                self.epsilon = 1. - final_num / len(self.src_point)
                self.calculate_N()
                final_src_point = self.src_point[point_ok[0], :]
                final_dst_point = self.dst_point[point_ok[0], :]

        return final_num, final_src_point, final_dst_point

    def my_transform(self, point, H):
        point = np.array([point[0], point[1], 1], dtype=np.float)
        result = H.dot(point)
        result = result / result[2]
        return result[:2]
