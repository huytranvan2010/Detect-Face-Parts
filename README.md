Trong bài trước chúng ta đã học được cách phát hiện các facial landmarks tren khuôn mặt. Trong bài này chúng ta sẽ đi trích xuất các vùng của khuôn mặt dựa vào facial landmarks.

**Face landmark detector** trong dlib tạo ta 68 tọa độ (x,y) của các facial landmarks (trong Python sẽ được đánh số từ 0 đến 67 - trên hình vẽ thường để từ 1 đến 68). 68 điểm này có được nhờ huấn luyện **shape predictor** trên bộ dữ liệu [iBUG 300-W dataset](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/).

<img src="https://www.pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup-768x619.jpg">

*Vị trí các facial landmarks*

Nhận thấy một số vùng trên khuôn mặt có thể tiếp cận được thông qua các chỉ số:
* Mouth [48, 68]
* Right eyebrow [17, 22]
* Left eyebrow [22, 27]
* Right eye [36, 42]
* Left eye [42, 48]
* Nose [27, 35]
* Jaw [0, 17]

Mapping này được hiện qua `FACIAL_LANDMARKS_IDXS` ordered dictionary bên trong `face_imutils` của thư viện `imutils`. Trong file `helper_functions.py` biểu diễn hai hàm tiện ích đã được build trong thư viện `imutils` để dễ hiểu hơn.

**Các thư viện sử dụng**:
* numpy
* dlib
* opencv
* imutils

https://pypi.org/project/opencv-python/ 

**Cách chạy**
```python
python detect_face_parts.py --shape_predictor shape_predictor_68_face_landmarks.dat --image input_image.png
```


