from google.colab import drive
import zipfile
import os


# Define paths to the zip files
zip_files = [
    '/content/drive/MyDrive/DATASET/Bn Data/Automatic_Annotation.zip',
    '/content/drive/MyDrive/DATASET/Bn Data/BN-HTR_Dataset.zip'
]

# Unzip the files
for zip_file in zip_files:
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall('/content/dataset')

# Path to the extracted dataset
data_dir = '/content/dataset'



import os
from PIL import Image
import torchvision.transforms as transforms

# Define transforms for ViT
vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT standard size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale normalization
])

def load_images(root_dir):
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jpg"):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Example usage
line_images = load_images(os.path.join(data_dir, "BN-HTR_dataset/Segmentation_Images/Lines"))


def load_annotations(root_dir):
    text_mapping = {}
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".txt"):
                doc_id = os.path.basename(root)
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                text_mapping[doc_id] = text
    return text_mapping

annotations = load_annotations(os.path.join(data_dir, "BN-HTR_dataset/Recognition_Ground_Truth_Texts"))



import os

def load_images(root_dir):
    image_paths = []
    print(f"Searching for images in: {root_dir}")  # ADD THIS LINE
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jpg"):  # Collect only .jpg files
                image_paths.append(os.path.join(root, file))
    return image_paths

# Path to the extracted dataset
data_dir = '/content/dataset'

# Corrected path
line_images_dir = os.path.join(data_dir, "BN-HTR_Dataset/Segmentation_Images/Lines")
line_images = load_images(line_images_dir)

# Verify the results
print("Number of line images:", len(line_images))
print("First 5 line images:", line_images[:5])



from sklearn.model_selection import train_test_split

# Split into train (80%), val (10%), and test (10%)
train_images, test_images = train_test_split(line_images, test_size=0.2, random_state=42)
val_images, test_images = train_test_split(test_images, test_size=0.5, random_state=42)

print("Train images:", len(train_images))
print("Validation images:", len(val_images))
print("Test images:", len(test_images))


#Build PyTorch Dataset

from torch.utils.data import Dataset

class BnHTRDataset(Dataset):
    def __init__(self, image_paths, text_mapping, transform=None):
        self.image_paths = image_paths
        self.text_mapping = text_mapping
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        doc_id = os.path.basename(img_path).split('_')[0]  # Extract document ID
        text = self.text_mapping.get(doc_id, "")
        tokens = tokenizer.tokenize(text)
        
        if self.transform:
            image = self.transform(image)
            
        return image, tokens
    

#Use Hugging Face transformers for ViT:

from transformers import ViTForImageClassification, TrainingArguments, Trainer

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(tokenizer.vocab),  # Adjust based on tokenizer
    ignore_mismatched_sizes=True
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    num_train_epochs=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Start training
trainer.train()