import os
import csv
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split

# Load Dataset from hugging face
dataset = load_dataset('alkzar90/NIH-Chest-X-ray-dataset', 'image-classification', trust_remote_code=True)

# Combine train and test split for custom train/test stratification
full_dataset = concatenate_datasets([dataset['train'], dataset['test']])
# Split all images into no_finding and finding using the first label (0 is the No Finding label, [0] index is used since all multi-label images have findings)
no_finding = full_dataset.filter(lambda x: x['labels'][0] == 0)
finding = full_dataset.filter(lambda x: x['labels'][0] != 0)
# Balance dataset by randomly selecting 51759 of the 60361 No Finding images to be included into the balanced dataset 
no_finding_sampled = no_finding.shuffle(seed=0).select(range(len(finding)))
balanced_dataset = concatenate_datasets([no_finding_sampled, finding]).shuffle(seed=0)

def add_binary_label(data):
    # Return 0 if first label is No Finding, else there is a Finding therefore return 1
    data['binary_label'] = 0 if data['labels'][0] == 0 else 1
    return data
# Add binary label
balanced_dataset = balanced_dataset.map(add_binary_label)

# Stratify keys based on the first label of each image
stratify_keys = [data['labels'][0] for data in tqdm(balanced_dataset, desc="Generating stratify keys")]
# Split dataset 80/20 for train test split
train_indices, test_indices = train_test_split(range(len(balanced_dataset)), test_size=0.2, stratify=stratify_keys, random_state=0)
# Create train and test data sets
train_dataset = balanced_dataset.select(train_indices)
test_dataset = balanced_dataset.select(test_indices)

# Define paths for images and csvs files
train_path = "/home/jimmy/dl/binary_model/NIH-Images/train"
test_path = "/home/jimmy/dl/binary_model/NIH-Images/test"
train_csv_path = "/home/jimmy/dl/binary_model/NIH-Images/train_labels.csv"
test_csv_path = "/home/jimmy/dl/binary_model/NIH-Images/test_labels.csv"
#Make sure paths exist
os.makedirs(train_path, exist_ok= True)
os.makedirs(test_path, exist_ok= True)

# Now that folders have been created, the images from each data set must be saved in the folders
# This can be done using a function 
def save_images(dataset, directory_path):
    # Iterate through the index of the dataset
    # Enumerate the set so that each image can have a unique name ie. 1.png 
    for idx, dtset in tqdm (enumerate(dataset), total = len(dataset), desc=f"Saving {directory_path}"):
        #Load image a resize it to 224x224 for DenseNet121
        img = dtset['image'].resize((224, 224))
        #Convert to RGB for DenseNet121
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Create the image path 
        img_path = os.path.join(directory_path, f"{idx}.png")
        # save the image using the path
        img.save(img_path)

# Save labels to csvs for processing and use within the CNN model
# This will be done in similar way to image saving
def save_labels(dataset, directory_path):
    # Open csv in write mode
    with open(directory_path, mode='w', newline='') as file: 
        writer = csv.writer(file)
        # Write row labels/names
        writer.writerow(["image_filename", "labels", "binary_label"])
        # iterate through the dataset
        # enumerate again so that each label will be paired with correct image
        for idx, example in tqdm(enumerate(dataset), total=len(dataset), desc=f"Saving {directory_path}"):
            # Set the image filename = name of images in folder
            img_name = f"{idx}.png"
            # Get labels from dataset
            labels = example['labels']
            binary_label = example['binary_label']
            # Write label and name to csv
            writer.writerow([img_name, labels, binary_label])

# Using function, save images
save_images(train_dataset, train_path)
save_images(test_dataset, test_path)

# Use the function to fill out the csv
save_labels(train_dataset, train_csv_path)
save_labels(test_dataset, test_csv_path)
