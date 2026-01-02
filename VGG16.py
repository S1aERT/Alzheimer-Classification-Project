import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping  # Import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight  # Import class_weight from sklearn
from PIL import Image

# Dataset Extraction
zip_path = r"C:\Users\sarth\Downloads\AugmentedAlzheimerDataset"
extract_path = "dataset"

if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

print("Dataset extracted successfully!")

# Data Preparation
DATA_DIR = os.path.join(extract_path, "Data")  
CATEGORIES = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very Mild Dementia']
IMG_SIZE = 64 

# Load and preprocess images
def load_data():
    data, labels = [], []
    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category)
        label = CATEGORIES.index(category)
        if not os.path.exists(path):
            print(f"Warning: Folder {path} not found!")
            continue
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append(img_array)
                labels.append(label)
            except Exception:
                pass
    return np.array(data, dtype=np.float32), np.array(labels)  

data, labels = load_data()
data = np.stack([cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in data]) / 255.0  
labels = keras.utils.to_categorical(labels, num_classes=len(CATEGORIES))

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
class_weight_dict = dict(enumerate(class_weights))

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Transfer Learning with VGG16
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(len(CATEGORIES), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# EarlyStopping Callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Model Training with class weights and Early Stopping
history = model.fit(datagen.flow(X_train, y_train, batch_size=16),
                    epochs=50,  # You can increase epochs
                    validation_data=(X_test, y_test),
                    class_weight=class_weight_dict,  # Apply class weights here
                    callbacks=[early_stopping])  # Add early stopping callback

# Save the new model
model.save('alzheimers_cnn_model_with_class_weights_and_early_stopping.h5')

# Plot Accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print("Model training complete.")
