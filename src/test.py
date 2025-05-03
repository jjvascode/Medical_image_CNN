import numpy as np
import pandas as pd
from pathlib import Path
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Read labels from csv and create dataframe for testing
test_frame = pd.read_csv("/mnt/c/Users/jjvas/OneDrive/Desktop/DL-Project/data/NIH-Images/test_label.csv")

# Keep track of folder path it will be used to join with img names to get full paths for each image
test_folder_path = "/mnt/c/Users/jjvas/OneDrive/Desktop/DL-Project/data/NIH-Images/test"

# Change dataframe so that image filename is the fullpath of the image
# lambda function will concat image filename with the folder path to create full path for images
test_frame.loc[:, ('image_filename')] = test_frame['image_filename'].apply(lambda x: os.path.join(test_folder_path, x))

# Labels are stored as strs, in order to manipulate frame as needed, they need to be convereted to ints
# Remove [] from str and convert to an int, lambda function: if x is a str it is convereted if not it remains x (int)
test_frame["labels"] = test_frame["labels"].apply(lambda x: int(x.strip("[]")) if isinstance(x, str) else x)

# The frame must be altered further need to add binary classification so 0 if label ==0 and 1 if label >0
# if the value of labels is not 0, then the value is set to 1 (0 is normal and 1 is anomaly detected)
test_frame["binary_label"] = (test_frame["labels"] != 0).astype(int)

# For illness classification 'labels' must be 1 hot encoded 
# This can be done using 
y = test_frame["labels"].values
one_hot = to_categorical(y, num_classes =15)
test_frame["illness"] = list(one_hot)

print(test_frame)
