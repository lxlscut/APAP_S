import copy

import cv2
import numpy as np

from APAP.apap import Ap
from UTIL.blending import blending_average
from UTIL.draw import Draw
from UTIL.glabal_homography import GloablHomography
from UTIL.image_info import Image_info
from UTIL.ransac import RANSAC
from UTIL.system_param import SystemParam
from UTIL.warp import warp_global, warp_local_homography_point
from feature.match import Match

if __name__ == '__main__':
    # todo 获取参数
    premeter = SystemParam()
    width = premeter.get_param("MESH_WIDTH")
    height = premeter.get_param("MESH_HEIGHT")
    # todo 获取图片
    src = cv2.imread("image/DSC00318.JPG")
    dst = cv2.imread("image/DSC00319.JPG")
    # todo 获取特征点
    match = Match(src, dst)
    match.getInitialFeaturePairs()
    src_point = match.src_match
    dst_point = match.dst_match

    # todo opencv 的方法求取全局矩阵
    H = cv2.findHomography(src_point, dst_point, cv2.RANSAC)

    draw = Draw()
    match = draw.draw_match(src=src, dst=dst, src_point=src_point, dst_point=dst_point)
    cv2.imshow("match", match)

    # # todo 对特征点进行ransac筛选
    ransac = RANSAC(src_point=src_point, dst_point=dst_point, threshold=3)
    final_num, final_src_point, final_dst_point = ransac.ransac()
    print(final_num)
    cal_src = copy.deepcopy(final_src_point)
    cal_dst = copy.deepcopy(final_dst_point)
    # todo 自己的方法求取矩阵
    gh = GloablHomography()
    gh_src = copy.deepcopy(final_src_point)
    gh_dst = copy.deepcopy(final_dst_point)
    h = gh.get_global_homo(gh_src, gh_dst)

    image_info = Image_info()

    image_info.get_final_size(src_img=src, dst_img=dst, H=h)
    result = warp_global(img=src, H=h, image_info=image_info)

    bg = np.zeros_like(result)
    bg[image_info.offset_y:src.shape[0] + image_info.offset_y, image_info.offset_x:src.shape[1] + image_info.offset_x,
    :] = dst[:, :, :]
    result = blending_average(bg, result)

    # apap 方法
    x = np.linspace(0, image_info.width, width + 1)
    y = np.linspace(0, image_info.height, height + 1)

    x_ = x + image_info.width / (width * 2)
    y_ = y + image_info.height / (height * 2)

    x_, y_ = np.meshgrid(x_[:width], y_[:height])
    vertices = np.stack((x_, y_), axis=-1)

    mesh = np.row_stack([x, y])
    vertices = vertices[:, :, :2] - np.array([image_info.offset_x, image_info.offset_y])

    a = Ap()
    cal_src_point = copy.deepcopy(final_src_point)
    csl_dst_point = copy.deepcopy(final_dst_point)
    local_h, local_weight = a.apap_stitch(src_point=cal_src_point, dst_point=csl_dst_point, vertices=vertices)

    warp_local = warp_local_homography_point(src=src, image_info=image_info, local_h=local_h, point=mesh)

    vertices = vertices.reshape((-1, 2))

    bg = np.zeros_like(warp_local)
    bg[image_info.offset_y:src.shape[0] + image_info.offset_y, image_info.offset_x:src.shape[1] + image_info.offset_x,
    :] = dst[:, :, :]
    warp_local = blending_average(warp_local, bg)
    # draw.draw(warp_local,vertices)

    cv2.imshow("warp_local_homography", warp_local)

    src_with_point = draw.draw(src, final_src_point)
    dst_with_point = draw.draw(dst, final_dst_point)
    cv2.imshow("src", src_with_point)
    cv2.imshow("dst", dst_with_point)
    cv2.imshow("result", result)
    cv2.waitKey(0)
