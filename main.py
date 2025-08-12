# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

# import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# %%
# Test image
from PIL import Image

img = Image.open('img.png')
print(img.format)

# %%
# Use forward slashes for paths
earthquake = os.listdir("disaster-images-dataset/Comprehensive Disaster Dataset(CDD)/Damaged_Infrastructure/Earthquake")
infrastructure = os.listdir("disaster-images-dataset/Comprehensive Disaster Dataset(CDD)/Damaged_Infrastructure/Infrastructure")
urban_fire_disaster = os.listdir("disaster-images-dataset/Comprehensive Disaster Dataset(CDD)/Fire_Disaster/Urban_Fire")
wild_fire_disaster = os.listdir("disaster-images-dataset/Comprehensive Disaster Dataset(CDD)/Fire_Disaster/Wild_Fire")
human_damage = os.listdir("disaster-images-dataset/Comprehensive Disaster Dataset(CDD)/Human_Damage")
drought = os.listdir("disaster-images-dataset/Comprehensive Disaster Dataset(CDD)/Land_Disaster/Drought")
landslide = os.listdir("disaster-images-dataset/Comprehensive Disaster Dataset(CDD)/Land_Disaster/Land_Slide")
water_disaster = os.listdir("disaster-images-dataset/Comprehensive Disaster Dataset(CDD)/Water_Disaster")

print(earthquake[0:5])
print(earthquake[-5:])

# %%
earthquake_label = [1]*36
infrastructure_label = [2]*1418
urban_fire_disaster_label = [3]*419
wild_fire_disaster_label = [4]*514
human_damage_label = [5]*240
drought_label = [6]*201
landslide_label = [7]*456
water_disaster_label = [8]*1035

labels = (earthquake_label + infrastructure_label + urban_fire_disaster_label + 
          wild_fire_disaster_label + human_damage_label + drought_label + 
          landslide_label + water_disaster_label)

print(len(labels))

# %%
earthquake_path = "disaster-images-dataset/Comprehensive Disaster Dataset(CDD)/Damaged_Infrastructure/Earthquake"
infrastructure_path = "disaster-images-dataset/Comprehensive Disaster Dataset(CDD)/Damaged_Infrastructure/Infrastructure"
urban_fire_disaster_path = "disaster-images-dataset/Comprehensive Disaster Dataset(CDD)/Fire_Disaster/Urban_Fire"
wild_fire_disaster_path = "disaster-images-dataset/Comprehensive Disaster Dataset(CDD)/Fire_Disaster/Wild_Fire"
human_damage_path = "disaster-images-dataset/Comprehensive Disaster Dataset(CDD)/Human_Damage"
drought_path = "disaster-images-dataset/Comprehensive Disaster Dataset(CDD)/Land_Disaster/Drought"
landslide_path = "disaster-images-dataset/Comprehensive Disaster Dataset(CDD)/Land_Disaster/Land_Slide"
water_disaster_path = "disaster-images-dataset/Comprehensive Disaster Dataset(CDD)/Water_Disaster"

# %%
# Image Processing
data = []
ImageFile.LOAD_TRUNCATED_IMAGES = True

for disaster_list, path in [
    (earthquake, earthquake_path),
    (infrastructure, infrastructure_path),
    (urban_fire_disaster, urban_fire_disaster_path),
    (wild_fire_disaster, wild_fire_disaster_path),
    (human_damage, human_damage_path),
    (drought, drought_path),
    (landslide, landslide_path),
    (water_disaster, water_disaster_path),
]:
    for img_file in disaster_list:
        # Safely open each image, resize, and convert to RGB
        image = Image.open(os.path.join(path, img_file))
        image = image.resize((128, 128))
        image = image.convert('RGB')
        image = np.array(image)
        data.append(image)

print("Total images processed:", len(data))

# %%
X = np.array(data)
Y = np.array(labels)

# do train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# %%
print(X.shape, X_train.shape, X_test.shape)

# %%
# Scale the data
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

# %%
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# %%
num_of_classes = 9
model = Sequential()

model.add(keras.Input(shape=(128,128,3)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_of_classes, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_scaled, Y_train, validation_split=0.2, epochs=10)

# %%
# Prediction test on 'urban.jpeg'
input_image_path = 'urban.jpeg'
input_image = cv2.imread(input_image_path)

input_image_resized = cv2.resize(input_image, (128,128))
input_image_scaled = input_image_resized / 255.0
input_image_reshaped = np.reshape(input_image_scaled, [1,128,128,3])

input_prediction = model.predict(input_image_reshaped)
print(input_prediction)

input_pred_label = np.argmax(input_prediction)
print("Predicted label:", input_pred_label)

