import numpy as np
from collections import OrderedDict
import cv2

""" 
	Thực chất 2 hàm này đã được xây dựng bên trong thư viện imutils.
	Cùng tìm hiểu xem nó hoạt động ra sao
"""

def shape_to_np(shape, dtype="int"):
	"""
		Hàm chuyển từ shape về numpy array (68 tọa độ facial landmarks)
		output là 2d numpy array có shape là (68, 2)
	"""
	# khởi tạo 2d-numpy array
	coords = np.zeros((68, 2), dtype=dtype)		# có 68 điểm
	# duyệt qua 68 facial landmarks và lấy các tọa độ
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)		# lấy tọa độ từng facial landmark
	# trả về 2d-numpy array
	return coords

FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mounth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("rigth_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])

# hàm vẽ các vùng trên khuôn mặt dựa trên facial landmarks
def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    """ 
		shape ở đây là 2d numpy array tọa độ 68 facial landmarks 
		output là ảnh đã được vẽ các face regions được xác định thông qua facial landmarks
	"""
    # tạo 2 bản copy  của ảnh ban đầu - một cái cho overlay, một cái cho output image
    overlay = image.copy()
    output = image.copy()

    # nếu colors=None (ko truyền vào gì), khởi tạo list of colors cho các vùng
    colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23), (168, 100, 168),
                (158, 163, 32), (163, 38, 32), (180, 42, 220)]

    # duyệt qua các facial landmark regions
    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        # lấy chỉ số đầu cuối của face regions
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        points = shape[j:k]     # không bao bồm chỉ số cuối

        # kiểm tra xem cs=ó phải là "jaw", chỉ cần vẽ đường hở nên làm riêng
        # vẽ đường đi qua các điểm
        if name == "jaw":
            for l in range(1, len(points)):
                pointA = tuple(points[l-1])     # vì mỗi một hàng trả về 2 giá trị, bt ở array nhưng chuyển về tuple để tí còn vẽ trong OpenCV
                pointB = tuple(points[l])       # chuyển luôn array về tuple các tọa độ
                cv2.line(overlay, pointA, pointB, colors[i], 2)     # tham số cuối là thichness
        # còn lại các cùng khác
        else:
            hull = cv2.convexHull(points)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)    # tham số cuối là thickness -1 tô luôn bên trong, tham số thứ 3 chỉ contour được vẽ, -1 vẽ tất cả
    
    # áp dụng transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)	# tham số cuối là dst để lưu ảnh đầu ra, output = overlay * alpha + output * (1-alpha)

    # return the output image
    return output