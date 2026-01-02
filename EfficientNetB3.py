import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
import zipfile
from PIL import Image
import io
import os
import matplotlib.pyplot as plt
from collections import Counter


from google.colab import drive
drive.mount('/content/drive')


ZIP_PATH = "/content/drive/MyDrive/PBL/dataset.zip"
IMG_SIZE = 224
BATCH_SIZE = 16
CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
EPOCHS = 100


def inspect_zip_structure(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        print("🔍 First 10 files in your zip:")
        for file in zip_file.namelist()[:10]:
            print(file)


def create_datasets(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_file:

        inspect_zip_structure(zip_path)


        all_files = []
        class_counts = {cls: 0 for cls in CLASSES}

        for f in zip_file.namelist():

            if f.startswith('dataset/') and any(f'dataset/{cls}' in f for cls in CLASSES):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):

                    for cls in CLASSES:
                        if f'dataset/{cls}' in f:
                            class_counts[cls] += 1
                    all_files.append(f)

        print(f"Class distribution in dataset: {class_counts}")
        print(f"Total image files found: {len(all_files)}")

        if len(all_files) == 0:
            raise ValueError("No image files found in the zip file. Please check the path and structure.")


        np.random.shuffle(all_files)
        split_idx = int(0.8 * len(all_files))
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]

        print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")


        def get_label(file):
            for cls_idx, cls_name in enumerate(CLASSES):
                if f'dataset/{cls_name}' in file:
                    return cls_idx
            print(f"Warning: Could not determine class for {file}")
            return 0

        train_labels = [get_label(f) for f in train_files]
        val_labels = [get_label(f) for f in val_files]


        label_counts = Counter(train_labels)
        total_samples = len(train_labels)
        n_classes = len(CLASSES)

        class_weight_dict = {}
        for class_idx in range(n_classes):
            count = label_counts.get(class_idx, 0)
            weight = total_samples / (n_classes * count) if count > 0 else 1.0
            class_weight_dict[class_idx] = weight

        print(f"Computed class weights: {class_weight_dict}")


        def generator(files, labels):
            with zipfile.ZipFile(zip_path, 'r') as zip_f:
                for file, label in zip(files, labels):
                    try:
                        with zip_f.open(file) as f:
                            img = Image.open(io.BytesIO(f.read()))
                            img = img.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
                            img = np.array(img) / 255.0
                        yield img, label
                    except Exception as e:
                        print(f"Error loading {file}: {str(e)}")
                        yield np.zeros((IMG_SIZE, IMG_SIZE, 3)), label


        train_ds = tf.data.Dataset.from_generator(
            lambda: generator(train_files, train_labels),
            output_signature=(
                tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_generator(
            lambda: generator(val_files, val_labels),
            output_signature=(
                tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


    return train_ds, val_ds, class_weight_dict, len(train_files), len(val_files)


print("Creating datasets...")
try:
    train_ds, val_ds, class_weights, train_files_count, val_files_count = create_datasets(ZIP_PATH)
    print("Datasets created successfully!")


    train_steps = max(1, train_files_count // BATCH_SIZE)
    val_steps = max(1, val_files_count // BATCH_SIZE)

except Exception as e:
    print(f"Error creating datasets: {str(e)}")
    raise


print("Building EfficientNetB3 model...")
base_model = EfficientNetB3(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)


for layer in base_model.layers[-20:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(CLASSES), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print("Model built and compiled!")


callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-6),
    ModelCheckpoint('/content/drive/MyDrive/PBL/best_model.h5', save_best_only=True)
]


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    class_weight=class_weights,
    callbacks=callbacks
)


final_loss, final_acc = model.evaluate(val_ds)
print(f"\n Final Validation Accuracy: {final_acc:.4f}")


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Progression')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Progression')
plt.legend()
plt.show()
print("Training complete!")
