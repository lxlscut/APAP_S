import numpy as np
import math
import copy
import cv2

from UTIL.system_param import SystemParam


class Ap:
    def __init__(self):
        self.gama, self.sigma = self.get_param()

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

    def apap_stitch(self, src_point, dst_point, vertices):
        src_cal = copy.deepcopy(src_point)
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
        simgma_inv = 1. / (self.sigma ** 2)
        A = np.zeros([cf1.shape[0] * 2, 9], dtype=np.float)
        locals_h = np.zeros([vertices.shape[0], vertices.shape[1], 3, 3], dtype=np.float)
        local_weight = np.zeros([vertices.shape[0], vertices.shape[1], src_point.shape[0]])
        for i in range(vertices.shape[0]):
            for j in range(vertices.shape[1]):
                for k in range(src_point.shape[0]):
                    # todo distance
                    distance = vertices[i, j] - src_cal[k]
                    weight = max(self.gama, math.exp(-(math.sqrt(distance[0] ** 2 + distance[1] ** 2) * simgma_inv)))
                    local_weight[i, j, k] = weight
                    A[2 * k, 0] = weight * cf1[k, 0]
                    A[2 * k, 1] = weight * cf1[k, 1]
                    A[2 * k, 2] = weight * 1
                    A[2 * k, 6] = weight * (-cf2[k, 0]) * cf1[k, 0]
                    A[2 * k, 7] = weight * (-cf2[k, 0]) * cf1[k, 1]
                    A[2 * k, 8] = weight * (-cf2[k, 0])

                    A[2 * k + 1, 3] = weight * cf1[k, 0]
                    A[2 * k + 1, 4] = weight * cf1[k, 1]
                    A[2 * k + 1, 5] = weight * 1
                    A[2 * k + 1, 6] = weight * (-cf2[k, 1]) * cf1[k, 0]
                    A[2 * k + 1, 7] = weight * (-cf2[k, 1]) * cf1[k, 1]
                    A[2 * k + 1, 8] = weight * (-cf2[k, 1])
                W, U, V = cv2.SVDecomp(A)
                h = V[-1, :]
                h = h.reshape((3, 3))
                h = np.linalg.inv(C2).dot(h).dot(C1)
                h = np.linalg.inv(N2).dot(h).dot(N1)
                h = h / h[2, 2]
                locals_h[i, j] = h

        return locals_h,local_weight

    def my_transform(self, point, H):
        point = np.array([point[0], point[1], 1], dtype=np.float)
        result = H.dot(point)
        result = result / result[2]
        return result[:2]
