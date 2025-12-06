import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os, json

# -------------------------------
# Directories
# -------------------------------
train_dir = os.path.join('dataset')

# -------------------------------
# Image data generator
# -------------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # MobileNetV2 input size
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# -------------------------------
# Base Model (MobileNetV2)
# -------------------------------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base layers (for transfer learning)
for layer in base_model.layers:
    layer.trainable = False

# -------------------------------
# Custom Layers for Cotton Disease Classification
# -------------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(train_data.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# -------------------------------
# Compile the model
# -------------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# Train the model
# -------------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

print("training accuracy:", history.history['accuracy'][-1])
# -------------------------------
# Save model and class indices
# -------------------------------
model.save('cotton_mobilenetv2_model.h5')

with open('class_indices.json', 'w') as f:
    json.dump(train_data.class_indices, f)

print("✅ Model training complete using MobileNetV2 and saved as cotton_mobilenetv2_model.h5")
