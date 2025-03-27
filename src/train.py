from datasets import load_dataset
from PIL import Image

#"image-classification", trust_remote_code = True

# Load Dataset from hugging face
dataset = load_dataset('alkzar90/NIH-Chest-X-ray-dataset', 'image-classification', trust_remote_code = True)

# Run print to test that data was loaded correctly. 
# Data is autsplit into train and test subsets
print(dataset['train'][:5])

# Create a data set for the train split 
train_data = dataset['train'][:5]

# Within the data you are given a PIL object for each "image" and labels
# These labels are enumerated from 0 to 12 (each being a different abnormality)
# Iterate through the lenght of train_data set
# at each iteration we will load the image at the ith index as well as teh label
# show each of the images using PIL .show() to ensure they are loaded properly
for i in range(len(train_data['image'])):
    image = train_data['image'][i]
    label = train_data['labels'][i]

    image.show()
