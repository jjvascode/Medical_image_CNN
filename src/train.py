from datasets import load_dataset
from PIL import Image

#"image-classification", trust_remote_code = True

# Load Dataset from hugging face
dataset = load_dataset('alkzar90/NIH-Chest-X-ray-dataset', 'image-classification', trust_remote_code = True)

# Run print to test that data was loaded correctly. 
# Data is autsplit into train and test subsets
print(dataset['train'][:5])

train_data = dataset['train'][:5]

for i in range(len(train_data['image'])):
    image = train_data['image'][i]
    label = train_data['labels'][i]

    image.show()
