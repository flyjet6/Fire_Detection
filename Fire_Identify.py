import numpy as np
import tensorflow as tf
from PIL import Image

target_size = (224, 224)

x = []
resized_images = []
for image_path in x:
    image = Image.open(image_path)
    image = image.resize(target_size)
    resized_images.append(np.array(image))

# Define the Sequential model
model = tf.keras.Sequential()

# Convolutional layers for feature extraction
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

# Flatten the feature maps
model.add(tf.keras.layers.Flatten())

# Fully connected layers
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit()
model.summary()