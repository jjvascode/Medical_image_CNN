import numpy as np
import pandas as pd
from pathlib import Path
import os
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D, Input, BatchNormalization, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import AUC

# Load in variables from .env
load_dotenv()

# Set variables needed
train_path = os.getenv("TRAIN_FOLDER_PATH")
train_csv = os.getenv("TRAIN_LABELS_CSV")

# Read labels from csv
# This will create dataframe
# this will have the image names as well as the labels
train_frame = pd.read_csv(train_csv)

# Keep track of folder paths 
# These will be used to join with img names to get full paths for each image
# The full paths will be needed in order to process the images into the CNN
train_folder_path = train_path


# Change dataframe so that image filename is the fullpath of the image
# for all hte rows, want to change the file name
# lambda function will concat image filename with the folder path to create full path for images
train_frame.loc[:, ('image_filename')] = train_frame['image_filename'].apply(lambda x: os.path.join(train_folder_path, x))


# Labels are stored as strs, in order to manipulate frame as needed, they need to be convereted to ints
# Remove [] from str and convert to an int, lambda function: if x is a str it is convereted if not it remains x (int)
train_frame["labels"] = train_frame["labels"].apply(lambda x: int(x.strip("[]")) if isinstance(x, str) else x)
# Add a binary label
# if the value of labels is not 0, then the value is set to 1 (0 is normal and 1 is anomaly detected)
train_frame["binary_label"] = (train_frame["labels"] != 0).astype(str)

print(train_frame.head)


# Preprocess images using keras generator adding augmentation
# Add rotation, zoom, height and width shifts as well as flips 
training_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    # Rescale pixel value to normalize values between 0 and 1
    rescale = 1./255,
    validation_split = .2, # Create a validation split
    rotation_range = 20,
    zoom_range = 0.15,
)

"""
    Once the generators are created, we want to load the images into them for testing 
    this can be done using flow_from_dataframe using the train and test frames
    dataframe = Set the dataframe from which to load images
    X-col will be path to images (or column that holds these in the df)
    y-col are the labels associated with each image
    target_size = Declare size of the image when processed
    batch size is set within generator
    class_mode = sets the class mode, in this case categorical (will take labels and 1 hot encode them as they are processed)
    subset= define subset (in this case it will be training and validation)
"""

training_images = training_gen.flow_from_dataframe(
    dataframe = train_frame,
    x_col = "image_filename",
    y_col = "binary_label",
    target_size = (256, 256),
    batch_size = 64, 
    class_mode = "binary",
    subset = "training", 
    shuffle=True
)

validation_images = training_gen.flow_from_dataframe(
    dataframe = train_frame,
    x_col = "image_filename",
    y_col = "binary_label",
    target_size = (256, 256),
    batch_size = 64, 
    class_mode = "binary",
    subset = "validation", 
    shuffle = False
)

# Build out model, using Conv2D layers and maxpooling
model = Sequential([
    Input(shape=(256, 256, 3)),
    Conv2D(32, (3,3), activation = 'relu', strides=1, padding = "same"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation = 'relu', strides=1, padding = "same"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation = 'relu', strides=1, padding = "same"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation = 'relu', strides=1, padding = "same"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(256, (3,3), activation = 'relu', strides=1, padding = "same"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # GlobalPool output that is to be fed to the connected dense layers
    Flatten(),
    
    # Dense layers using dropout
    Dense(256, activation = 'relu'),
    Dropout(0.2),
    Dense(128, activation = 'relu'),
    Dropout(0.2),
    Dense(64, activation = 'relu'),
    Dropout(0.2),

    # Output layer will be 15 nodes for illness classification and use softmax activation
    Dense(1, activation = 'sigmoid')
])


# Compile model using adam optimzer, binary crossentropy as the loss (binary classification) and measure accuracy
model.compile(
    optimizer = 'adam',
    loss = "binary_crossentropy",
    metrics =["accuracy", AUC(name='auc')]
)

# Can use summary to check and ensure model was created properly
model.summary()

# Implement callbacks and LROn to improve learning and stop if learning plateaus
callback =[
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(filepath="/mnt/c/Users/jjvas/OneDrive/Desktop/DL-Project/models/binary_model.keras", monitor="val_loss", save_best_only =True),
    ReduceLROnPlateau(monitor="val_loss", factor=.2, patience=3, min_lr=1e-6)
]


# Run model after ensuring it is created properly 
history = model.fit(
    training_images, 
    validation_data = validation_images,
    epochs = 25, 
    callbacks=callback
)

# Plot the results of the trainings using matplotlib
# Want to plot the 2 values that were tested, accuracy vs epoch and loss vs epoch
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')
plt.savefig("training_plot.png")
print("Plot saved as training_plot.png")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Epochs')
plt.savefig("training_loss_plot.png")
print("Plot saved as training_loss_plot.png")

plt.plot(history.history['auc'], label='Training AUC')
plt.plot(history.history['val_auc'], label='Validation AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()
plt.title('AUC Over Epochs')
plt.savefig("training_auc_plot.png")
print("Plot saved as training_auc_plot.png")
