import os
import torch
import json
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import ViTModel
import torch.nn as nn
from collections import defaultdict
from sklearn.model_selection import train_test_split

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_eot = False  # End-of-token marker

class BnGraphemizer:
    def __init__(self, vocabulary=None):
        self.root = TrieNode()
        if vocabulary:
            self.build_trie(vocabulary)
        
        # Bengali vowels (স্বরবর্ণ)
        self.vowels = set([
            'অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ'
        ])

        # Bengali consonants (ব্যঞ্জনবর্ণ)
        self.consonants = set([
            'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ',
            'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ',
            'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ', 'ড়', 'ঢ়', 'য়', 'ৎ'
        ])

        # Vowel diacritics (কার)
        self.diacritics = set([
            'া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ', '্'
        ])

    def build_trie(self, vocabulary: list):
        """Build trie from the grapheme vocabulary."""
        for grapheme in vocabulary:
            node = self.root
            for char in grapheme:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_eot = True  # Mark end of grapheme

    def tokenize(self, sequence: str) -> list:
        """Tokenize a sequence into graphemes using the trie."""
        # If we have a vocabulary-based trie, use it
        if self.root.children:
            return self._tokenize_with_trie(sequence)
        
        # Otherwise use character-based tokenization with Bengali rules
        return self._tokenize_chars(sequence)
    
    def _tokenize_with_trie(self, sequence: str) -> list:
        """Tokenize using the trie (vocabulary-based approach)"""
        tokens = []
        i = 0
        while i < len(sequence):
            node = self.root
            j = i
            last_eot_pos = -1
            
            while j < len(sequence) and sequence[j] in node.children:
                node = node.children[sequence[j]]
                j += 1
                if node.is_eot:
                    last_eot_pos = j
            
            if last_eot_pos != -1:
                tokens.append(sequence[i:last_eot_pos])
                i = last_eot_pos
            else:
                tokens.append(sequence[i])
                i += 1
        
        return tokens
    
    def _tokenize_chars(self, sequence: str) -> list:
        """Tokenize based on character rules for Bengali"""
        tokens = []
        i = 0
        while i < len(sequence):
            # Check for consonant + hasant + consonant (conjunct)
            if (i + 2 < len(sequence) and 
                sequence[i] in self.consonants and 
                sequence[i+1] == '্' and 
                sequence[i+2] in self.consonants):
                # Add the conjunct (3 characters)
                tokens.append(sequence[i:i+3])
                i += 3
            # Check for consonant + vowel diacritic
            elif (i + 1 < len(sequence) and 
                  sequence[i] in self.consonants and 
                  sequence[i+1] in self.diacritics):
                # Add consonant + diacritic together
                tokens.append(sequence[i:i+2])
                i += 2
            else:
                # Add single character
                tokens.append(sequence[i])
                i += 1
        
        return tokens

def load_vocabulary(file_path: str) -> list:
    """Load grapheme vocabulary from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # Check if the loaded data is a dictionary and contains "anscombe" key
            if isinstance(data, dict) and "anscombe" in data:
                return data["anscombe"]
            elif isinstance(data, list):
                # If it's a list directly return it
                return data
            else:
                print("JSON file does not contain expected structure. Using character-based tokenization.")
                return []
    except Exception as e:
        print(f"Error loading vocabulary: {e}. Using character-based tokenization.")
        return []

def load_annotations(root_dir):
    """
    Load text annotations from Recognition_Ground_Truth_Texts directory.
    The structure is expected to be:
    - root_dir/
      - 1/
        - 1.txt (contains ground truth for document 1)
      - 2/
        - 2.txt (contains ground truth for document 2)
      ...
    
    Returns a dictionary mapping document IDs to their text content.
    """
    text_mapping = {}
    print(f"Loading annotations from: {root_dir}")
    
    try:
        # Navigate through all document folders
        for doc_id in os.listdir(root_dir):
            doc_path = os.path.join(root_dir, doc_id)
            
            # Skip if not a directory
            if not os.path.isdir(doc_path):
                continue
            
            # Look for the .txt file named the same as the folder
            txt_file = os.path.join(doc_path, f"{doc_id}.txt")
            if os.path.exists(txt_file):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    text_mapping[doc_id] = text
                    print(f"Loaded annotation for document {doc_id}")
        
        print(f"Total annotations loaded: {len(text_mapping)}")
    except Exception as e:
        print(f"Error loading annotations: {e}")
    
    return text_mapping

def load_line_images(lines_dir):
    """
    Load line image paths from the Segmentation_Images/Lines directory.
    The structure is expected to be:
    - lines_dir/
      - 1/
        - 1_1.jpg (first line of document 1)
        - 1_2.jpg (second line of document 1)
      - 2/
        - 2_1.jpg (first line of document 2)
        ...
    
    Returns a dictionary mapping document IDs to lists of line image paths.
    """
    document_lines = defaultdict(list)
    print(f"Loading line images from: {lines_dir}")
    
    try:
        # Navigate through all document folders
        for doc_id in os.listdir(lines_dir):
            doc_path = os.path.join(lines_dir, doc_id)
            
            # Skip if not a directory
            if not os.path.isdir(doc_path):
                continue
            
            # Collect all line images for this document
            for file in os.listdir(doc_path):
                if file.endswith(".jpg"):
                    img_path = os.path.join(doc_path, file)
                    document_lines[doc_id].append(img_path)
            
            print(f"Loaded {len(document_lines[doc_id])} lines for document {doc_id}")
        
        print(f"Total documents with line images: {len(document_lines)}")
    except Exception as e:
        print(f"Error loading line images: {e}")
    
    return document_lines

class BnHTRDataset(Dataset):
    def __init__(self, document_lines, annotations, tokenizer, token_to_id, transform=None):
        """
        Dataset for Bengali handwritten text recognition.
        
        Args:
            document_lines: Dictionary mapping document IDs to lists of line image paths
            annotations: Dictionary mapping document IDs to ground truth text
            tokenizer: Tokenizer for Bengali text
            token_to_id: Mapping from tokens to IDs
            transform: Image transformations
        """
        self.transform = transform
        self.tokenizer = tokenizer
        self.token_to_id = token_to_id
        
        # Create flat list of (image_path, doc_id) pairs for valid documents
        self.samples = []
        for doc_id, img_paths in document_lines.items():
            # Skip documents without annotations
            if doc_id not in annotations:
                print(f"Warning: Document {doc_id} has no annotation")
                continue
            
            # Add all lines for this document
            for img_path in img_paths:
                self.samples.append((img_path, doc_id))
        
        print(f"Dataset initialized with {len(self.samples)} valid line images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, doc_id = self.samples[idx]
        
        try:
            # Load and process image
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            
            if self.transform:
                image = self.transform(image)
                
            # Get the ground truth text for this document
            text = annotations[doc_id]
            
            # Tokenize the text and convert to token IDs
            tokens = self.tokenizer.tokenize(text)
            token_ids = [self.token_to_id.get(token, self.token_to_id['<UNK>']) for token in tokens]
            
            return image, torch.tensor(token_ids)
            
        except Exception as e:
            print(f"Error processing sample {img_path}: {e}")
            # Return default values in case of error
            dummy_image = torch.zeros(3, 224, 224) if self.transform else Image.new('L', (224, 224))
            return dummy_image, torch.tensor([self.token_to_id['<UNK>']])

class ViTForBengaliOCR(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.classifier = nn.Linear(self.vit.config.hidden_size, vocab_size)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        sequence_output = outputs.last_hidden_state  # Shape: (batch_size, num_patches, hidden_size)
        logits = self.classifier(sequence_output)  # Shape: (batch_size, num_patches, vocab_size)
        return logits

def collate_fn(batch):
    """
    Collate function for DataLoader that handles variable-length sequences.
    Returns:
        images: Tensor of shape (batch_size, channels, height, width)
        padded_ids: Tensor of shape (batch_size, max_seq_len) with padded token IDs
        target_lengths: Tensor of shape (batch_size,) with original sequence lengths
    """
    # Filter out any None samples
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None, None
    
    images = [item[0] for item in batch]
    token_ids = [item[1] for item in batch]
    
    # Get original sequence lengths
    target_lengths = torch.tensor([len(ids) for ids in token_ids])
    
    # Stack images
    images = torch.stack(images)
    
    # Pad token sequences
    padded_ids = pad_sequence(token_ids, batch_first=True, padding_value=token_to_id['<PAD>'])
    
    return images, padded_ids, target_lengths

def train_epoch(model, dataloader, optimizer, ctc_loss, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        if batch[0] is None:  # Skip empty batches
            continue
            
        images, token_ids, target_lengths = batch
        images = images.to(device)
        token_ids = token_ids.to(device)
        target_lengths = target_lengths.to(device)
        
        # Forward pass
        logits = model(images)  # Shape: (batch_size, num_patches, vocab_size)
        batch_size, seq_length, vocab_size = logits.size()
        
        # Apply log softmax to get log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)
        
        # Set input lengths (all sequences have the same length = number of patches)
        input_lengths = torch.full((batch_size,), seq_length, dtype=torch.long).to(device)
        
        # Compute CTC loss
        try:
            loss = ctc_loss(
                log_probs.permute(1, 0, 2),  # (time, batch, vocab) - CTC expects time-first
                token_ids,
                input_lengths,
                target_lengths
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress
            if num_batches % 10 == 0:
                print(f"Batch {num_batches}, Loss: {loss.item():.4f}")
            
        except Exception as e:
            print(f"Error in CTC loss calculation: {e}")
            print(f"Batch size: {batch_size}, Sequence length: {seq_length}")
            print(f"Target lengths min: {target_lengths.min().item()}, max: {target_lengths.max().item()}")
            continue
    
    return total_loss / max(1, num_batches)

def evaluate(model, dataloader, ctc_loss, device):
    """Evaluate the model on validation/test data."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if batch[0] is None:
                continue
                
            images, token_ids, target_lengths = batch
            images = images.to(device)
            token_ids = token_ids.to(device)
            target_lengths = target_lengths.to(device)
            
            # Forward pass
            logits = model(images)
            batch_size, seq_length, vocab_size = logits.size()
            
            # Apply log softmax
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            
            # Set input lengths
            input_lengths = torch.full((batch_size,), seq_length, dtype=torch.long).to(device)
            
            try:
                loss = ctc_loss(
                    log_probs.permute(1, 0, 2),
                    token_ids,
                    input_lengths,
                    target_lengths
                )
                total_loss += loss.item()
                num_batches += 1
            except Exception as e:
                print(f"Error in validation: {e}")
                continue
    
    return total_loss / max(1, num_batches)

def decode_predictions(logits, id_to_token):
    """
    Decode the model's predictions using greedy decoding.
    
    Args:
        logits: Tensor of shape (batch_size, seq_length, vocab_size)
        id_to_token: Mapping from token IDs to tokens
        
    Returns:
        List of decoded texts
    """
    # Get the most likely token IDs
    _, predictions = torch.max(logits, dim=2)  # shape: (batch_size, seq_length)
    
    # Convert to list of lists
    predictions = predictions.cpu().numpy().tolist()
    
    # Decode predictions
    decoded_texts = []
    for pred in predictions:
        # Greedy decoding - collapse repeated tokens and remove blanks
        collapsed = []
        prev = -1
        for p in pred:
            if p != prev and p != 0:  # 0 is <PAD> token (blank)
                collapsed.append(p)
            prev = p
        
        # Convert IDs to tokens
        text = ''.join([id_to_token[token_id] for token_id in collapsed])
        decoded_texts.append(text)
    
    return decoded_texts

def main():
    # Path to your dataset
    data_dir = '/content/dataset'
    
    # Define paths for the dataset components
    annotations_dir = os.path.join(data_dir, "BN-HTR_Dataset/Recognition_Ground_Truth_Texts")
    lines_dir = os.path.join(data_dir, "BN-HTR_Dataset/Segmentation_Images/Lines")
    
    # Check for alternative paths if needed
    if not os.path.exists(annotations_dir):
        annotations_dir = os.path.join(data_dir, "BN-HTR_dataset/Recognition_Ground_Truth_Texts")
    
    if not os.path.exists(lines_dir):
        lines_dir = os.path.join(data_dir, "BN-HTR_dataset/Segmentation_Images/Lines")
    
    print(f"Using annotations directory: {annotations_dir}")
    print(f"Using line images directory: {lines_dir}")
    
    # Check if directories exist
    if not os.path.exists(annotations_dir) or not os.path.exists(lines_dir):
        print("Error: Required directories not found!")
        return
    
    # Load grapheme vocabulary if available
    vocab_path = os.path.join(data_dir, "graphemes.json")
    vocabulary = load_vocabulary(vocab_path) if os.path.exists(vocab_path) else []
    
    # Initialize tokenizer
    tokenizer = BnGraphemizer(vocabulary)
    
    # Load annotations and line images
    annotations = load_annotations(annotations_dir)
    document_lines = load_line_images(lines_dir)
    
    # Check if we have data
    if not annotations or not document_lines:
        print("Error: No data loaded!")
        return
    
    # Build vocabulary from annotations
    print("Building vocabulary from annotations...")
    all_tokens = set()
    all_tokens.add('<PAD>')  # ID 0 for padding/blank
    all_tokens.add('<UNK>')  # ID 1 for unknown tokens
    
    for doc_id, text in annotations.items():
        tokens = tokenizer.tokenize(text)
        all_tokens.update(tokens)
    
    # Create token mappings
    vocab = sorted(list(all_tokens))
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    id_to_token = {idx: token for idx, token in enumerate(vocab)}
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Data transforms for ViT
    vit_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.expand(3, -1, -1)),  # Expand grayscale to 3 channels
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create flat list of document IDs for splitting
    all_doc_ids = list(document_lines.keys())
    
    # Split documents into train/val/test (60%/20%/20%)
    train_docs, test_docs = train_test_split(all_doc_ids, test_size=0.4, random_state=42)
    val_docs, test_docs = train_test_split(test_docs, test_size=0.5, random_state=42)
    
    print(f"Train documents: {len(train_docs)}")
    print(f"Validation documents: {len(val_docs)}")
    print(f"Test documents: {len(test_docs)}")
    
    # Create document-based datasets
    train_lines = {doc_id: document_lines[doc_id] for doc_id in train_docs if doc_id in document_lines}
    val_lines = {doc_id: document_lines[doc_id] for doc_id in val_docs if doc_id in document_lines}
    test_lines = {doc_id: document_lines[doc_id] for doc_id in test_docs if doc_id in document_lines}
    
    # Create datasets
    train_dataset = BnHTRDataset(train_lines, annotations, tokenizer, token_to_id, transform=vit_transform)
    val_dataset = BnHTRDataset(val_lines, annotations, tokenizer, token_to_id, transform=vit_transform)
    test_dataset = BnHTRDataset(test_lines, annotations, tokenizer, token_to_id, transform=vit_transform)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8,  # Reduced batch size to avoid memory issues
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=8, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Initialize model
    model = ViTForBengaliOCR(len(vocab))
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    
    # Define CTC loss
    ctc_loss = torch.nn.CTCLoss(blank=token_to_id['<PAD>'], reduction='mean')
    
    # Training parameters
    num_epochs = 10
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, ctc_loss, device)
        
        # Validate
        val_loss = evaluate(model, val_loader, ctc_loss, device)
        
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'vocab': vocab,
                'token_to_id': token_to_id,
                'id_to_token': id_to_token
            }, 'best_bn_ocr_model.pth')
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
    
    print("Training completed!")
    
    # Load best model for evaluation
    checkpoint = torch.load('best_bn_ocr_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    test_loss = evaluate(model, test_loader, ctc_loss, device)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Sample inference
    model.eval()
    with torch.no_grad():
        for images, _, _ in test_loader:
            if images is None:
                continue
            
            images = images.to(device)
            
            # Get predictions
            logits = model(images)
            
            # Decode predictions
            decoded_texts = decode_predictions(logits, id_to_token)
            
            # Print sample results
            for i, text in enumerate(decoded_texts[:3]):  # Show first 3 examples
                print(f"Sample {i+1}: {text}")
            
            break  # Only process one batch

if __name__ == "__main__":
    main()