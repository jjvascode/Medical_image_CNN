import numpy as np
import pandas as pd
from pathlib import Path
import os
import tensorflow as tf
import keras
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# Load in variables from .env
load_dotenv()

# Set variables needed
train_path = os.getenv("TRAIN_FOLDER_PATH")
train_csv = os.getenv("TRAIN_LABELS_CSV")

# Read labels from both csvs
# This will create dataframe
# this will have the image names as well as the labels
train_frame = pd.read_csv(train_csv)

# Keep track of folder paths 
# These will be used to join with img names to get full paths for each image
# The full paths will be needed in order to process the images into the CNN
train_folder_path = train_path


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


print(train_frame)

# Create a train and validation split using scikit learn
# Want to stratify as data within the set is skewed resulting in a larger proportion of "normal" images
train_df, val_df = train_test_split(train_frame, test_size=0.2, stratify=train_frame["binary_label"], random_state=42)




"""
The built in ImageDataGenerator from Keras while good, does not allow for multi-label classification
in order to deal with this, the use of a custom generator was needed.THe base of the code was based on this 
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly. Where it discusses the use of custom generators using keras Sequence. 
This use of a custom generator allows for binary classification (normal vs not normal) as well as multi-class classification (0-14) where each is 
a different illness.
"""

class CustomGen(Sequence):
    def __init__(self, dataframe, batch_size = 64, img_size =(256, 256), shuffle=True):
        # Initialization
        self.dataframe = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.dataframe))
        self.on_epoch_end()

    # len function will denote the number of batches per epoch
    def __len__(self):
        return int(np.ceil(len(self.dataframe)/ self.batch_size))
    
    # Getthe batch of the indices
    def __getitem__(self, index):
        # generate the indexes of the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1)*self.batch_size]
        batch_df = self.dataframe.iloc[batch_indices]

        # Initialize arrays
        X = np.zeros((len(batch_df), *self.img_size, 3), dtype=np.float32)
        y_binary = np.zeros((len(batch_df), 1), dtype=np.float32)
        y_illness = np.zeros((len(batch_df), 15), dtype=np.float32)

        # Process each image
        for i, row in enumerate(batch_df.itertuples()):
            img = load_img(row.image_filename, target_size=self.img_size)
            img = img_to_array(img) / 255.0  # Normalize to [0, 1]
            X[i] = img
            y_binary[i] = row.binary_label
            y_illness[i] = np.array(row.illness)

        return X, {"binary_output": y_binary, "illness_output": y_illness}
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

input_image = Input(shape=(256,256,3))

# Conv layers using MaxPooling and GlobalAvgPooling
x = Conv2D(32, (3,3), activation ='relu')(input_image)
x2 = MaxPooling2D(2,2)(x)
x3 = Conv2D(64, (3,3), activation ='relu')(x2)
x4 = MaxPooling2D(2,2)(x3)
x5 = Conv2D(126, (3,3), activation='relu')(x4)
x6 = MaxPooling2D(2,2)(x5)
x7 = Conv2D(256, (3,3), activation='relu')(x6)
x8 = MaxPooling2D(2,2)(x7)
x9 = GlobalAveragePooling2D()(x8)

# Dense Layers using Dropout
Dense_1 = Dense(256, activation = 'relu')(x9)
Drop_1 = Dropout(.3)(Dense_1)
Dense_2 = Dense(128, activation = 'relu')(Drop_1)
Drop_2 = Dropout(.3)(Dense_2)
Dense_3 = Dense(64, activation = 'relu')(Drop_2)
Drop_3 = Dropout(0.3)(Dense_3)

# Output Layers
binary_output = Dense(1, activation='sigmoid', name ="binary_output")(Drop_3)
illness_output = Dense(15, activation='softmax', name ="illness_output")(Drop_3)

#Create model 
model = Model(inputs = input_image, outputs=[binary_output, illness_output])
model.compile(optimizer=Adam(), 
            loss={"binary_output": "binary_crossentropy",
                  "illness_output": "categorical_crossentropy"},
            metrics={
                "binary_output": "accuracy",
                "illness_output": "accuracy"
            }
)


# Can use summary to check and ensure model was created properly
#model.summary()

# Create needed dataframes, one for training and the other for validation
train_gen = CustomGen(train_df, batch_size=64, img_size=(256,256), shuffle=True)
val_gen = CustomGen(val_df, batch_size=64, img_size=(256,256), shuffle=False)

model.fit(train_gen, validation_data = val_gen, epochs=10)


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