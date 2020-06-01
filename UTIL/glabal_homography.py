from UTIL.system_param import SystemParam
import copy
import numpy as np
import math
import cv2


class GloablHomography:
    def __init__(self):
        pass

    def get_param(self):
        system = SystemParam()
        self.gama = system.get_param("gama")
        self.sigma = system.get_param("sigma")
        return self.gama, self.sigma

    def getNormalize2DPts(self, point):
        """
        :param point: [num,2]
        :return:
        """
        origin_point = copy.deepcopy(point)
        padding = np.ones(point.shape[0], dtype=np.float)
        # todo 求均值
        c = np.mean(point, axis=0)
        point[:, :2] = point[:, :2] - c[:2]
        squre = np.square(point)
        sum = np.sum(squre, axis=1)
        mean = np.mean(np.sqrt(sum))
        scale = math.sqrt(2) / mean
        t = np.array([[scale, 0, -scale * c[0]],
                      [0, scale, -scale * c[1]],
                      [0, 0, 1]], dtype=np.float)
        origin_point = np.column_stack((origin_point, padding))
        new_point = t.dot(origin_point.T)
        new_point = new_point.T[:, :2]
        return t, new_point

    def getConditionerFromPts(self, point):
        calculate = np.expand_dims(point, 0)
        mean_pts, std_pts = cv2.meanStdDev(calculate)
        mean_pts = np.squeeze(mean_pts)
        std_pts = np.squeeze(std_pts)
        std_pts = std_pts * std_pts * point.shape[0] / (point.shape[0] - 1)
        std_pts = np.sqrt(std_pts)
        std_pts[0] = std_pts[0] + (std_pts[0] == 0)
        std_pts[1] = std_pts[1] + (std_pts[1] == 0)
        T = np.array([[math.sqrt(2) / std_pts[0], 0, (-math.sqrt(2) / std_pts[0] * mean_pts[0])],
                      [0, math.sqrt(2) / std_pts[1], (-math.sqrt(2) / std_pts[1] * mean_pts[1])],
                      [0, 0, 1]], dtype=np.float)

        return T

    def get_global_homo(self, src_point, dst_point):

        N1, nf1 = self.getNormalize2DPts(src_point)
        N2, nf2 = self.getNormalize2DPts(dst_point)

        C1 = self.getConditionerFromPts(nf1)
        C2 = self.getConditionerFromPts(nf2)

        cf1, cf2 = np.zeros_like(nf1), np.zeros_like(nf2)
        for i in range(cf1.shape[0]):
            cf1[i, 0] = nf1[i, 0] * C1[0, 0] + C1[0, 2]
            cf1[i, 1] = nf1[i, 1] * C1[1, 1] + C1[1, 2]
            cf2[i, 0] = nf2[i, 0] * C2[0, 0] + C2[0, 2]
            cf2[i, 1] = nf2[i, 1] * C2[1, 1] + C2[1, 2]

        # todo 构建A矩阵
        A = np.zeros([src_point.shape[0] * 2, 9], dtype=np.float)
        for k in range(src_point.shape[0]):
            # todo distance
            A[2 * k, 0] = cf1[k, 0]
            A[2 * k, 1] = cf1[k, 1]
            A[2 * k, 2] = 1
            A[2 * k, 6] = (-cf2[k, 0]) * cf1[k, 0]
            A[2 * k, 7] = (-cf2[k, 0]) * cf1[k, 1]
            A[2 * k, 8] = (-cf2[k, 0])

            A[2 * k + 1, 3] = cf1[k, 0]
            A[2 * k + 1, 4] = cf1[k, 1]
            A[2 * k + 1, 5] = 1
            A[2 * k + 1, 6] = (-cf2[k, 1]) * cf1[k, 0]
            A[2 * k + 1, 7] = (-cf2[k, 1]) * cf1[k, 1]
            A[2 * k + 1, 8] = (-cf2[k, 1])
        W, U, V = cv2.SVDecomp(A)
        h = V[-1, :]
        h = h.reshape((3, 3))
        h = np.linalg.inv(C2).dot(h).dot(C1)
        h = np.linalg.inv(N2).dot(h).dot(N1)
        h = h / h[2, 2]
        return h

    def my_transform(self, point, H):
        point = np.array([point[0], point[1], 1], dtype=np.float)
        result = H.dot(point)
        result = result / result[2]
        return result[:2]
