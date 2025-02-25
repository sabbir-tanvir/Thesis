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



from torch.utils.data import Dataset

class BnHTRDataset(Dataset):
    def __init__(self, image_paths, text_mapping, tokenizer, token_to_id, transform=None):
        self.image_paths = image_paths
        self.text_mapping = text_mapping
        self.tokenizer = tokenizer
        self.token_to_id = token_to_id
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale

        # Extract document ID (e.g., "1" from "1_1.jpg")
        doc_id = os.path.basename(img_path).split('_')[0]
        text = self.text_mapping.get(doc_id, "")

        # Tokenize and convert to IDs
        tokens = self.tokenizer.tokenize(text)
        token_ids = [self.token_to_id.get(token, self.token_to_id['<UNK>']) for token in tokens]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(token_ids)

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

from transformers import ViTModel
import torch.nn as nn

class ViTForBengaliOCR(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.classifier = nn.Linear(self.vit.config.hidden_size, vocab_size)  # Output per-patch logits

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        sequence_output = outputs.last_hidden_state  # Shape: (batch_size, num_patches, hidden_size)
        logits = self.classifier(sequence_output)  # Shape: (batch_size, num_patches, vocab_size)
        return logits
    


from torch.utils.data import Dataset

class BnHTRDataset(Dataset):
    def __init__(self, image_paths, text_mapping, tokenizer, token_to_id, transform=None):
        self.image_paths = image_paths
        self.text_mapping = text_mapping
        self.tokenizer = tokenizer
        self.token_to_id = token_to_id
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale

        # Extract document ID (e.g., "1" from "1_1.jpg")
        doc_id = os.path.basename(img_path).split('_')[0]
        text = self.text_mapping.get(doc_id, "")

        # Tokenize and convert to IDs
        tokens = self.tokenizer.tokenize(text)
        token_ids = [self.token_to_id.get(token, self.token_to_id['<UNK>']) for token in tokens]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(token_ids)
    


from transformers import ViTModel
import torch.nn as nn

class ViTForBengaliOCR(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.classifier = nn.Linear(self.vit.config.hidden_size, vocab_size)  # Output per-patch logits

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        sequence_output = outputs.last_hidden_state  # Shape: (batch_size, num_patches, hidden_size)
        logits = self.classifier(sequence_output)  # Shape: (batch_size, num_patches, vocab_size)
        return logits

from torch.utils.data import Dataset

class BnHTRDataset(Dataset):
    def __init__(self, image_paths, text_mapping, tokenizer, token_to_id, transform=None):
        self.image_paths = image_paths
        self.text_mapping = text_mapping
        self.tokenizer = tokenizer
        self.token_to_id = token_to_id
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale

        # Extract document ID (e.g., "1" from "1_1.jpg")
        doc_id = os.path.basename(img_path).split('_')[0]
        text = self.text_mapping.get(doc_id, "")

        # Tokenize and convert to IDs
        tokens = self.tokenizer.tokenize(text)
        token_ids = [self.token_to_id.get(token, self.token_to_id['<UNK>']) for token in tokens]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(token_ids)




from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    images = [item[0] for item in batch]
    token_ids = [item[1] for item in batch]

    # Pad token sequences
    padded_ids = pad_sequence(token_ids, batch_first=True, padding_value=token_to_id['<PAD>'])

    # Stack images
    images = torch.stack(images)

    return images, padded_ids




from collections import defaultdict

# Initialize tokenizer
tokenizer = BnGraphemizer()

# Collect all graphemes from text annotations
vocab = defaultdict(int)
for text in annotations.values():
    tokens = tokenizer.tokenize(text)
    for token in tokens:
        vocab[token] += 1

# Create token-to-ID mapping
vocab = ['<PAD>', '<UNK>'] + sorted(vocab.keys())
token_to_id = {token: idx for idx, token in enumerate(vocab)}
id_to_token = {idx: token for idx, token in enumerate(vocab)}

print("Vocabulary size:", len(vocab))


# Create datasets
train_dataset = BnHTRDataset(train_images, annotations, tokenizer, token_to_id, transform=vit_transform)
val_dataset = BnHTRDataset(val_images, annotations, tokenizer, token_to_id, transform=vit_transform)
test_dataset = BnHTRDataset(test_images, annotations, tokenizer, token_to_id, transform=vit_transform)

from torch.utils.data import DataLoader

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)




import torch
from torch.optim import Adam

# Initialize model
model = ViTForBengaliOCR(len(vocab))

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize optimizer
optimizer = Adam(model.parameters(), lr=1e-4)

# Define CTC loss function
ctc_loss = torch.nn.CTCLoss(blank=token_to_id['<PAD>'])



# Create datasets
train_dataset = BnHTRDataset(train_images, annotations, tokenizer, token_to_id, transform=vit_transform)



# Example: Check the shape of the first image
image, _ = train_dataset[0]
print("Image shape:", image.shape)  # Should be (3, 224, 224)

def train_epoch(model, dataloader, optimizer, ctc_loss, device):
    model.train()
    total_loss = 0
    for images, token_ids in dataloader:
        images = images.to(device)
        token_ids = token_ids.to(device)

        # Forward pass
        logits = model(images)  # Shape: (batch_size, num_patches, vocab_size)

        # Compute CTC loss
        input_lengths = torch.full((images.size(0),), logits.size(1), dtype=torch.long).to(device)  # All patches
        target_lengths = torch.tensor([len(t) for t in token_ids], dtype=torch.long).to(device)

        loss = ctc_loss(
            logits.permute(1, 0, 2),  # (time, batch, vocab)
            token_ids,
            input_lengths,
            target_lengths,
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, ctc_loss, device)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}")



def decode_predictions(logits, id_to_token):
    # logits: (batch_size, num_patches, vocab_size)
    pred_ids = torch.argmax(logits, dim=2)
    pred_texts = []
    for batch in pred_ids:
        tokens = [id_to_token[id.item()] for id in batch if id != token_to_id['<PAD>']]
        pred_texts.append(''.join(tokens))
    return pred_texts

# Example inference
model.eval()
with torch.no_grad():
    images, _ = next(iter(test_loader))
    images = images.to(device)
    logits = model(images)
    pred_texts = decode_predictions(logits, id_to_token)

print("Predicted Texts:", pred_texts)