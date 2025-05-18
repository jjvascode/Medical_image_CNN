#Import libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #type: ignore
from tensorflow.keras.applications import DenseNet121 # type: ignore
from tensorflow.keras.applications.densenet import preprocess_input # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.utils import to_categorical, Sequence # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
from tensorflow.keras.models import load_model #type: ignore
# Read labels from both csvs
# This will create dataframe
# this will have the image names as well as the labels
train_frame = pd.read_csv("/home/jimmy/dl/binary_model/NIH-Images/train_labels.csv")
test_frame = pd.read_csv("/home/jimmy/dl/binary_model/NIH-Images/test_labels.csv")

# Keep track of folder paths 
# These will be used to join with img names to get full paths for each image
# The full paths will be needed in order to process the images into the CNN
train_folder_path = "/home/jimmy/dl/binary_model/NIH-Images/train"
test_folder_path = "/home/jimmy/dl/binary_model/NIH-Images/test"

# Change dataframe so that image filename is the fullpath of the image
# lambda function will concat image filename with the folder path to create full path for images
train_frame.loc[:, ('image_filename')] = train_frame['image_filename'].apply(lambda x: os.path.join(train_folder_path, x))
test_frame.loc[:, ('image_filename')] = test_frame['image_filename'].apply(lambda x: os.path.join(test_folder_path, x))

print(train_frame)
print(test_frame)

# train and validation split already created in datahandler.py
train_df = train_frame
val_df = test_frame

"""
The built in ImageDataGenerator from Keras while good, does not allow for multi-label classification
in order to deal with this, the use of a custom generator was needed.THe base of the code was based on this 
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly. Where it discusses the use of custom generators using keras Sequence. 
This use of a custom generator allows for binary classification (normal vs not normal) as well as multi-class classification (0-14) where each is 
a different illness.
"""

class CustomGen(Sequence):
    def __init__(self, dataframe, batch_size = 32, img_size =(224, 224), shuffle=True):
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
        y_binary = np.zeros((len(batch_df),), dtype=np.float32)

        # Process each image
        for i, row in enumerate(batch_df.itertuples()):
            img = load_img(row.image_filename, target_size=self.img_size)
            img = img_to_array(img)
            img = preprocess_input(img)
            X[i] = img
            y_binary[i] = row.binary_label

        return X, y_binary
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
# Create needed dataframes, one for training and the other for validation
train_gen = CustomGen(train_df, batch_size=32, img_size=(224,224), shuffle=True)
val_gen = CustomGen(val_df, batch_size=32, img_size=(224,224), shuffle=False)

#Define DenseNet121 as the base model, do not include model head, use imagenet weights, use normal input shape, and disable training the base model
base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False
#Define new model head and use GlobalAveragePooling2D() to reduce parameters and lower the risk of overfitting
x = GlobalAveragePooling2D()(base_model.output)
#Use batch normalization to normalize activations (make sure they have a mean of 0 and a standard deviation of 1)
x = BatchNormalization()(x)
#Implement dropout to lower risk of overfitting
x = Dropout(0.25)(x)
#Use a dense layer with 512 neurons and relu activation
x = Dense(512, activation='relu')(x)
#Implement another dropout layer to lower risk of overfitting
x = Dropout(0.5)(x)
#Implement an output layer with 1 neuron for binary classification
output = Dense(1, activation='sigmoid')(x)
#Define model to use DenseNet121 input layer and new output layer (output = Dense(15, activation='softmax')(x))
model = Model(inputs=base_model.input, outputs=output)
#Compile model, have it use Adam optimizer, binary_crossentropy loss (for binary classification), and have it measure accuracy
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
#Save model when it's at the epoch where it has the lowest validation loss 
best_model_callback = ModelCheckpoint(filepath='Initial_DenseNet_ChestXRay.keras', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
#Store training history for loss plot and train model for 5 epochs
history = model.fit(train_gen, validation_data=val_gen, epochs=5, callbacks=[best_model_callback])
#Generate loss plot
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("Initial_Loss_Plot.png")
print("Plot saved as Initial_Loss_Plot.png")

#Load model from first round of training for additional training, set compile to false to allow for unfreezing layers
model = load_model("Initial_Binary_ChestXRay.keras", compile=False)
#Unfreeze all layers after layer 91 of DenseNet121
for layer in model.layers[91:]:
    layer.trainable = True
#Compile model, have it use Adam optimizer, binary_crossentropy loss (for binary classification), and have it measure accuracy
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
#Save model when it's at the epoch where it has the lowest validation loss 
best_model_callback = ModelCheckpoint(filepath='Second_DenseNet_ChestXRay.keras', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
#Store training history for loss plot and train model for 10 epochs
history = model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=[best_model_callback])
#Generate loss plot
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("Second_Loss_Plot.png")
print("Plot saved as Second_Loss_Plot.png")

#Load model from second round of training for additional training, set compile to false to allow for unfreezing layers
model = load_model("Second_Binary_ChestXRay.keras", compile=False)
#Unfreeze all layers of DenseNet121
for layer in model.layers[61:]:
    layer.trainable = True
#Compile model, have it use Adam optimizer, set learning rate to 1e-4 to prevent overfitting, binary_crossentropy loss (for binary classification), and have it measure accuracy
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
#Save model when it's at the epoch where it has the lowest validation loss 
best_model_callback = ModelCheckpoint(filepath='Final_Binary_ChestXRay.keras', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
#Store training history for loss plot and train model for 10 epochs
history = model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=[best_model_callback])
#Evaluate final model based of accuracy
test_loss, test_acc = model.evaluate(val_gen)
print(f"Test Accuracy: {test_acc:.4f}")
#Generate loss plot
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("Final_Loss_Plot.png")
print("Plot saved as Final_Loss_Plot.png")
