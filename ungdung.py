import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# Khởi tạo các biến để đếm các dự đoán đúng và tổng
correct_preds = 0
count0 = 0
count1 = 0

# Tải mô hình
model = load_model('model.h5')
model.summary()

# Xác định nhãn lớp cho khẩu trang và không có khẩu trang
labels = ['Không đeo khẩu trang','Có đeo khẩu trang' ]

# Đường dẫn đến thư mục chứa ảnh
folder_paths = ['C:/nhandien/traindata/co deo', 'C:/nhandien/traindata/khong deo']

# Khởi tạo một danh sách trống để lưu trữ các dự đoán
preds = []

# Lặp qua các tệp hình ảnh trong mỗi thư mục
for folder_path, label in zip(folder_paths, [1, 0]):
    for image_file in os.listdir(folder_path):
        
        # Lấy đường dẫn đầy đủ đến tệp hình ảnh
        image_path = os.path.join(folder_path, image_file)
        
        # Nối một bộ chứa đường dẫn hình ảnh và nhãn vào danh sách đặt trước
        preds.append((image_path, label))
        
       
for pred in preds:
    print("Image path: {}, label: {}".format(pred[0], pred[1]))

# Lấy danh sách tất cả các tệp hình ảnh trong thư mục
image_files = os.listdir(folder_path)

# Chọn ngẫu nhiên một ảnh từ danh sách
random_image = random.choice(image_files)

# Yêu cầu người dùng nhập số lượng ảnh muốn hiển thị
num_images = int(input("Bạn muốn hiển thị bao nhiêu ảnh?: "))

# Chọn ngẫu nhiên số lượng ảnh từ danh sách
random_images = random.sample(image_files, num_images)

# Tạo đối tượng subplot với số hàng và cột tương ứng với số lượng hình ảnh
num_images = len(random_images)
rows = int(np.sqrt(num_images))
cols = int(np.ceil(num_images / rows))
fig, axs = plt.subplots(rows, cols, figsize=(cols*10, rows*10))

# Lặp lại tất cả các hình ảnh và thực hiện phát hiện khẩu trang
pred_values = []
pred_values1 = []

# Lặp lại tất cả các hình ảnh và thực hiện phát hiện khẩu trang
for i, (image_path, label) in enumerate(random.sample(preds, num_images)):
    # Tải hình ảnh bằng OpenCV
    image = cv2.imread(image_path)

    # Thay đổi kích thước hình ảnh khuôn mặt để phù hợp với kích thước đầu vào của mô hình
    face_resized = cv2.resize(image, (200, 200))

    # Bình thường hóa các giá trị pixel của hình ảnh khuôn mặt trong khoảng từ 0 đến 1
    face_normalized = np.expand_dims(face_resized, axis=0) / 255.0

    # Dự đoán khuôn mặt bằng mô hình
    pred = model.predict(face_normalized)
    
    # Nối giá trị dự đoán vào danh sách
    pred_values.append(pred.flatten()[0])  

    # In giá trị predict
    print("Giá trị pred: {}".format(np.array2string(pred.flatten(), separator=', ')))

    # làm tròn giá trị predict   
    pred_rounded = np.round(pred).astype(int)

    pred_values1.append(pred_rounded.flatten()[0]) 

    # in Giá trị pred_rounded
    print("Làm tròn giá trị predict: {}".format(np.array2string(pred_rounded.flatten(), separator=', ')))

    # So sánh giá trị dự đoán với giá trị thực tế và in kết quả
    if pred_rounded == label:
        print(f"Ảnh {i+1}: Dự đoán đúng như {labels[label]}")
        correct_preds += 1 
    else:
        print(f"Ảnh {i+1}: Dự đoán sai như {labels[pred_rounded[0][0]]}, nhãn thực tế là {labels[label]}")

    # Tính tổng số ảnh được dự đoán là 0
    for pred in pred_rounded:
        if pred == 0:
            count0 += 1

    # Tính tổng số ảnh được dự đoán là 1
    for pred in pred_rounded:
        if pred == 1:
            count1 += 1

        
    # Hiển thị hình ảnh trong một subplot
    if rows > 1 and cols > 1:
        row = i // cols
        col = i % cols
        axs[row, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        axs[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
# In kết quả Tổng số ảnh được dự đoán là 0 
print("Tổng số ảnh được dự đoán Không đeo khẩu trang là:", count0)

# In kết quả Tổng số ảnh được dự đoán là 1
print("Tổng số ảnh được dự đoán Có đeo khẩu trang là:", count1)

# In kết quả số ảnh được đoán đúng
print("Số ảnh được đoán đúng:", correct_preds)

# Tính toán tỷ lệ phần trăm ảnh được dự đoán đúng
num_images = len(random_images)
accuracy = (correct_preds / num_images) * 100

# In kết quả tỷ lệ phần trăm ảnh được dự đoán đúng
print("Tỷ lệ phần trăm ảnh được dự đoán đúng: {:.2f}%".format(accuracy))


# Vẽ biểu đồ kết quả giá trị pred
plt.figure()
plt.plot(range(len(pred_values)), pred_values, marker='o')
plt.plot(range(len(pred_values1)), pred_values1, marker='o')
plt.xlabel('Image')
plt.ylabel('Giá trị dự đoán')
plt.title('Giá trị dự đoán')
plt.xticks(range(len(pred_values)))

plt.show()

