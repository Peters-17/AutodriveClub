# 加载依赖库
# load all dependent libraries
import cv2
import numpy as np
# 编写主函数
# main function
if __name__ == "__main__":
    # 读取图片
    # read the raw picture and resize
    img = cv2.imread('./753fdd2f776b75fc47f8a509a932156.png')
    img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)
    # 在彩色图像的情况下，解码图像将以b g r顺序存储通道。
    # store tubes on the order of B,G,R
    grid_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 从RGB色彩空间转换到HSV色彩空间
    # transfer RGB to HSV
    grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)

    # since the color difference is big, we can use binary image to better extract the edge
    # H、S、V范围一：
    # range for HSV, to produce mask1
    # All values below is chosen based on the sheet of HSV colors and modified by trying for best fitting
    lower1 = np.array([0, 154, 20])
    upper1 = np.array([3, 255, 255])
    # mask1 为二值图像
    # mask1 is a binary image
    mask1 = cv2.inRange(grid_HSV, lower1, upper1)
    res1 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask1)

    # H、S、V范围二：
    # range for HSV, to produce mask2
    lower2 = np.array([179, 154, 50])
    upper2 = np.array([180, 255, 255])
    # mask2 为二值图像
    # mask2 is a binary image
    mask2 = cv2.inRange(grid_HSV, lower2, upper2)
    res2 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask2)

    # 将两个二值图像结果 相加
    # sum a new mask
    mask3 = mask1 + mask2
    # 生成mask三通道图
    # produce a RGB picture of mask
    color_mask_RGB = cv2.merge([mask3 * 255, mask3 * 255, mask3 * 255])

    imggray = cv2.cvtColor(color_mask_RGB, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imggray, 0, 255, cv2.THRESH_OTSU)
    # cv2.CHAIN_APPROX_NONE，所有的边界点都会被存储 all edgepoints will be stored here
    # cv2.CHAIN_APPROX_SIMPLE 只存储端点 only store the endpoints of the contour
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 目标轮廓
    # target contour
    target_contours = []
    # 目标点
    # target points
    points = []
    # 创建白色画布
    # create a white background
    white = np.ones((img.shape[0], img.shape[1], 3))
    # 遍历轮廓
    # iterate the contour
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 75: # 75 made by trying, can be other values, like 30, 50 or 100...
            target_contours.append(contour)
            # 查找质心
            # image moments
            mu = cv2.moments(contour, False)
            mc = [mu['m10'] / mu['m00'], mu['m01'] / mu['m00']]
            # 画圆
            # draw circle
            cv2.circle(white, (int(mc[0]), int(mc[1])), 3, [0,0,255], -1)
            points.append([int(mc[0]), int(mc[1])])

    # 目标点按照Y排序
    # sort by y axis
    points.sort(key = lambda x:x[1], reverse=False)
    # 区分左右
    # identify whether the point is left or right
    if points[-2][0] < points[-1][0]:
        point_ld = (points[-2][0],points[-2][1])
        point_rd = (points[-1][0],points[-1][1])
    else:
        point_ld = (points[-1][0],points[-1][1])
        point_rd = (points[-2][0],points[-2][1])
    # 获取中点X坐标
    # get the location of middle point xx
    xx = (point_ld[0] + point_rd[0])/2
    # 获取和中点在X方向上的绝对距离列表
    # create a list, increasingly stored the distance between a point and xx in x axis
    list_dis = []
    for p in points:
        list_dis.append([p[0],p[1],abs(p[0] - xx)])
    # 绝对距离排序
    list_dis.sort(key = lambda x:x[2], reverse=True)
    # 获取最上面两个点
    # 区分左右
    # identify whether the point is left or right
    if list_dis[-1][0] < list_dis[-2][0]:
        point_lu = (list_dis[-1][0],list_dis[-1][1])
        point_ru = (list_dis[-2][0], list_dis[-2][1])
    else:
        point_lu = (list_dis[-2][0], list_dis[-2][1])
        point_ru = (list_dis[-1][0], list_dis[-1][1])
    # 绘制目标线
    # debug
    print(point_lu)
    print(point_ld)
    # draw lines
    def draw_radial(img,p1,p2):
        x1 = 10000
        k = (p2[1] - p1[1]) / (p2[0] - p1[0])
        y1 = x1 * k

        x2 = -10000
        k = (p2[1] - p1[1]) / (p2[0] - p1[0])
        y2 = x2 * k

        cv2.line(img,p1,(int(p1[0]+x1),int(p1[1]+y1)),(0,255,0),2)
        cv2.line(img, p1, (int(p1[0] + x2), int(p1[1] + y2)), (0, 255, 0), 2)
        return img

    img = draw_radial(img,point_lu,point_ld)
    img = draw_radial(img, point_ru, point_rd)
    # 结果显示
    # show the result picture
    cv2.imshow("output", img)


    cv2.waitKey(0)
    cv2.destroyAllWindows()