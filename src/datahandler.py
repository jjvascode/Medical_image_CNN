
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import csv

#"image-classification", trust_remote_code = True

# Load Dataset from hugging face
dataset = load_dataset('alkzar90/NIH-Chest-X-ray-dataset', 'image-classification', trust_remote_code = True)


# Filter out data which may have multiple labels
def clean_data(data):
    # This will only return points with a single label
    return len(data['labels']) == 1

# Use filter to clean both the training and testing data
cleaned_train_data = dataset['train'].filter(clean_data)
cleaned_test_data = dataset['test'].filter(clean_data)

# Once the data has been cleaned, I want to create a new dict to store the new data
# Run print to test that data was loaded correctly. 
# Data is autsplit into train and test subsets
print(cleaned_train_data[0])
print(cleaned_test_data[0])

# Create a folder within data to store filtered images 
# Create folders within to house both test data and train data
save_path = "/mnt/c/Users/jjvas/OneDrive/Desktop/DL-Project/data/NIH-Images"
train_path = "/mnt/c/Users/jjvas/OneDrive/Desktop/DL-Project/data/NIH-Images/train"
test_path = "/mnt/c/Users/jjvas/OneDrive/Desktop/DL-Project/data/NIH-Images/test"

# Paths for the csv files
train_label_csv = "/mnt/c/Users/jjvas/OneDrive/Desktop/DL-Project/data/NIH-Images/train_labels.csv"
test_label_csv = "/mnt/c/Users/jjvas/OneDrive/Desktop/DL-Project/data/NIH-Images/test_label.csv"

# Ensure that the directories are made
os.makedirs(train_path, exist_ok= True)
os.makedirs(test_path, exist_ok= True)

# Now that folders have been created, the images from each data set must be saved in the folders
# This can be done using a function 
def save_images(dataset, directory_path):
    # Iterate through the index of the dataset
    # Enumerate the set so that each image can have a unique name ie. 1.png 
    for idx, dtset in tqdm (enumerate(dataset), total = len(dataset), desc="Saving Images"):
        # Extract the image from dataset
        img = dtset['image']

        #Convert the images to greyscale for sake of consistency 
        if img.mode == 'RGBA':
            img = img.convert('L')
        # Create the image path 
        img_path = os.path.join(directory_path, f"{idx}.png")
        # save the image using the path
        img.save(img_path)

# Using function, save images
#save_images(cleaned_train_data, train_path)
#save_images(cleaned_test_data, test_path)

# Save labels to csvs for processing and use within the CNN model
# This will be done in similar way to image saving 
def save_labels(dataset, csv_path):

    # Open csv in write mode
    with open(csv_path, mode='w', newline='') as file: 
        writer = csv.writer(file)

        # Write row labels/names
        writer.writerow(["image_filename", "labels"])

    # iterate through the dataset
    # enumerate again so that each label will be paired with correct image
        for idx, example in tqdm(enumerate(dataset), total=len(dataset), desc=f"Saving {csv_path}"):

            # Set the image filename = name of images in folder
            img_name = f"{idx}.png"

            # Get labels from dataset
            labels = example['labels']

            # Write label and name to csv
            writer.writerow([img_name, labels])

# Use the function to fill out the csv
save_labels(cleaned_train_data, train_label_csv)
save_labels(cleaned_test_data, test_label_csv)


# Create a data set for the train split 
#train_data = dataset['train'][1]

# Within the data you are given a PIL object for each "image" and labels
# These labels are enumerated from 0 to 12 (each being a different abnormality)
# Iterate through the lenght of train_data set
# at each iteration we will load the image at the ith index as well as teh label
# show each of the images using PIL .show() to ensure they are loaded properly
#for i in range(len(train_data['image'])):
#    image = train_data['image'][i]
#    label = train_data['labels'][i]

#    image.show()