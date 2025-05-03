import numpy as np
import pandas as pd
from pathlib import Path
import os
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Read labels from both csvs
# This will create dataframe
# this will have the image names as well as the labels
train_frame = pd.read_csv("/mnt/c/Users/jjvas/OneDrive/Desktop/DL-Project/data/NIH-Images/train_labels.csv")

# Keep track of folder paths 
# These will be used to join with img names to get full paths for each image
# The full paths will be needed in order to process the images into the CNN
train_folder_path = "/mnt/c/Users/jjvas/OneDrive/Desktop/DL-Project/data/NIH-Images/train"


# Change dataframe so that image filename is the fullpath of the image
# lambda function will concat image filename with the folder path to create full path for images
train_frame.loc[:, ('image_filename')] = train_frame['image_filename'].apply(lambda x: os.path.join(train_folder_path, x))


# Labels are stored as strs, in order to manipulate frame as needed, they need to be convereted to ints
# Remove [] from str and convert to an int, lambda function: if x is a str it is convereted if not it remains x (int)
train_frame["labels"] = train_frame["labels"].apply(lambda x: int(x.strip("[]")) if isinstance(x, str) else x)


# The frame must be altered further need to add binary classification o if label ==0 and 1 if label >0
# if the value of labels is not 0, then the value is set to 1 (0 is normal and 1 is anomaly detected)
train_frame["binary_label"] = (train_frame["labels"] != 0).astype(int)

# In order to do illness classification as well, label needs to be one hot encoded 
# This can be done using to_categorical
y = train_frame["labels"].values
hot_encoded = to_categorical(y, num_classes =15)
train_frame["illness"] = list(hot_encoded)

print(train_frame.dtypes)





"""
# Preprocess images using keras generator (first run will use not augmentation)
training_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    # Rescale pixel value to normalize values between 0 and 1
    rescale = 1./255,
    # Create a validation split
    validation_split = .2
)

# Create a gen for testing images as well 
test_gen = tf.keras.preprocessing.image.ImageDataGenerator (
    # Normalize 
    rescale = 1./255
)

# Once the generators are created, we want to load the images into them for testing 
# this can be done using flow_from_dataframe using the train and test frames
training_images = training_gen.flow_from_dataframe(
    # Set the dataframe from which to load images
    dataframe = train_frame,
    # X-col will be path to images
    x_col = "image_filename",
    #y-col are the labels,
    y_col = "binary_label",
    # Declare size of the image
    target_size = (256, 256),
    # batch size
    batch_size = 64, 
    # Class mode
    class_mode = "binary",
    # define subset
    subset = "training"
)

validation_images = training_gen.flow_from_dataframe(
    # Set the dataframe from which to load images
    dataframe = train_frame,
    # X-col will be path to images
    x_col = "image_filename",
    #y-col are the labels,
    y_col = "binary_label",
    # Declare size of the image
    target_size = (256, 256),
    # batch size
    batch_size = 64, 
    # Class mode
    class_mode = "binary",
    # define subset
    subset = "validation"
)

# Build out model, using Conv2D layers, maxpooling, etc. 
model = Sequential([
    Conv2D(32, (3,3), activation = 'relu', padding = "same", input_shape = (256, 256, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation = 'relu', padding = "same"),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation = 'relu', padding = "same"),
    MaxPooling2D(2,2),
    Conv2D(256, (3,3), activation = 'relu', padding = "same"),
    MaxPooling2D(2,2),

    # Flatten output that is to be fed to the connected dense layers
    GlobalAveragePooling2D(),
    
    Dense(256, activation = 'relu'),
    Dropout(0.5),
    Dense(128, activation = 'relu'),
    Dropout(0.5),
    Dense(64, activation = 'relu'),
    Dropout(0.5),

    # Output layer will be a single node and use sigmoid activation
    Dense(1, activation = 'sigmoid')
])

# Compile the model that was jsut made
# use adam optimzer, binary crossentropy as the loss (binary classification) and measure accuracy
model.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics =["accuracy"]
)

# Can use summary to check and ensure model was created properly
#model.summary()

# Implement callbacks and LROn to improve learning and stop if learning plateaus
callback =[
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    #ModelCheckpoint(filepath="/mnt/c/Users/jjvas/OneDrive/Desktop/DL-Project/models/best_model.h5", monitor="val_loss", save_best_only =True),
    ReduceLROnPlateau(monitor="val_loss", factor=.2, patience=3, min_lr=1e-6)
]

# Run model after ensuring it is created properly 
history = model.fit(
    training_images, 
    validation_data = validation_images,
    epochs = 25, 
    callbacks=callback
)
"""