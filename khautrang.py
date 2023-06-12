import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Dropout, Flatten
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

# Kích thước ảnh input
img_rows = 200
img_cols = 200

# Load dữ liệu từ thư mục
train_dir = 'C:/nhandien/traindata'

# Khởi tạo ImageDataGenerator để load dữ liệu từ thư mục
x_train = ImageDataGenerator(rescale=1./255)

train = x_train.flow_from_directory(train_dir, target_size=(img_rows, img_cols), batch_size=10, class_mode='binary')
print("Số lượng mẫu đào tạo:", train.n)

# Xác định mô hình
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 3)

model = keras.Sequential(
    [
        #Một lớp đầu vào với hình dạng của hình ảnh đầu vào được xác định bởi input_shape biến.
        keras.Input(shape=input_shape),

        #Một lớp tích chập với 32 bộ lọc tương ứng và kích thước hạt nhân là 3x3.
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),

        #Một lớp tổng hợp tối đa với kích thước nhóm là 2x2.
        layers.MaxPooling2D(pool_size=(2, 2)),

        #Một lớp tích chập với 64 bộ lọc tương ứng và kích thước hạt nhân là 3x3.
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),

        #Một lớp tổng hợp tối đa với kích thước nhóm là 2x2.
        layers.MaxPooling2D(pool_size=(2, 2)),

        #Một lớp tích chập với 128 bộ lọc tương ứng và kích thước hạt nhân là 3x3.
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),

         #Một lớp tổng hợp tối đa với kích thước nhóm là 2x2.
        layers.MaxPooling2D(pool_size=(2, 2)),

        #Một lớp Flatten để chuyển đổi đầu ra từ các lớp tích chập thành mảng 1D.
        layers.Flatten(),

        #Một lớp dropout  với tỷ lệ dropout  là 25% để giảm tình trạng thừa.
        layers.Dropout(0.25),

        layers.Dense(1, activation='sigmoid'),
    ]
)

model.summary()

#Mô hình được biên dịch với hàm binary cross-entropy, trình tối ưu hóa RMSprop và chỉ số độ chính xác.
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

# Train model
model.fit(train,epochs=20,validation_data=train)

#Lưu model
model.save('model.h5')

print("train hoàn tất")


history = model.history.history
# Hiển thị đồ thị accuracy và loss
plt.plot(history['accuracy'], label='accuracy')
plt.plot(history['val_accuracy'], label='val_accuracy')
plt.plot(history['loss'], label='loss')
plt.plot(history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.legend()

# Lưu đồ thị vào tệp tin hình ảnh
plt.savefig('C:/nhandien/dothianh.png')

plt.show()
