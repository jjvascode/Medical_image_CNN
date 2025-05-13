import numpy as np
import pandas as pd
from pathlib import Path
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from dotenv import load_dotenv

# Load in variables from .env
load_dotenv()

# Load the saved model
model = load_model("/mnt/c/Users/jjvas/OneDrive/Desktop/DL-Project/models/best_model.keras")

# Set variables needed
test_path = os.getenv("TEST_FOLDER_PATH")
test_csv = os.getenv("TEST_LABELS_CSV")

# Read labels from csv and create dataframe for testing
test_frame = pd.read_csv(test_csv)

# Keep track of folder path it will be used to join with img names to get full paths for each image
test_folder_path = test_path

# Change dataframe so that image filename is the fullpath of the image
# lambda function will concat image filename with the folder path to create full path for images
test_frame.loc[:, ('image_filename')] = test_frame['image_filename'].apply(lambda x: os.path.join(test_folder_path, x))

# Create a gen for testing images as well (Normalize images)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator (
    rescale = 1./255
)

# flow from datafram to load images into the gen.
test_data = test_gen.flow_from_dataframe(
    dataframe=test_frame,
    x_col='image_filename',
    y_col='labels',              
    target_size=(256, 256),     
    batch_size=32,
    class_mode='categorical',   
    shuffle=False               
)

model.summary()
model.evaluate(test_data)
