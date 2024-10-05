import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm


class IM2LatexDataset(Dataset):
    def __init__(self, transform=None, split='train'):

        # Load the dataset from huggingface
        print('Loading dataset.')
        self.dataset = load_dataset(
            'yuntian-deng/im2latex-100k',
            split=split,
        )

        # Convert dataset to iterable dataset
        # This way. all 50000 are not loaded into memory
        # Thus reducing lag and leads to faster looping
        iter_dataset = self.dataset.to_iterable_dataset()
        self.transform = transform
        self.length = 0

        # Create vocabulary
        self.vocab = set()
        print('Creating vocabulary.')
        for row in tqdm(iter_dataset, total=len(self.dataset)):
            self.length += 1
            self.vocab.update(row['formula'].strip())
        self.vocab = sorted(list(self.vocab))

        # Create dicts for encoding and decoding formulas
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get formula and the image from the dataset
        formula = self.dataset[idx]['formula'].strip()
        image = self.dataset[idx]['image'].convert('L')

        # Apply transform to image if needed
        if self.transform:
            image = self.transform(image)

        # Encode formula
        encoded_formula = [self.char_to_idx[char] for char in formula]

        return image, torch.tensor(encoded_formula)

    # def decode_formula(self, encoded_formula):
    #     return ''.join(
    #         [self.idx_to_char[idx.item()] for idx in encoded_formula]
    #     )

    #My changes
    def decode_formula(self, encoded_formula):
        """
        Decode either model outputs (logits) or encoded indices back into a formula string.
        
        Args:
            encoded_formula: Can be either:
                - A tensor of indices directly (1D tensor)
                - A tensor of logits (2D tensor: sequence_length x vocab_size)
        
        Returns:
            str: The decoded formula as a string
        """
        try:
            # Check if input is logits (2D tensor) or indices (1D tensor)
            if encoded_formula.dim() == 2:
                # If logits, convert to indices first
                indices = torch.argmax(encoded_formula, dim=-1)
            else:
                indices = encoded_formula

            # Convert indices to characters
            decoded_chars = []
            for idx in indices:
                try:
                    # Convert tensor to integer and look up in vocabulary
                    idx_val = idx.item() if torch.is_tensor(idx) else idx
                    if idx_val in self.idx_to_char:
                        char = self.idx_to_char[idx_val]
                        # Optional: Skip special tokens if you have them
                        if char not in ['<pad>', '<sos>', '<eos>']:
                            decoded_chars.append(char)
                except (KeyError, ValueError) as e:
                    print(f"Warning: Could not decode index {idx_val}: {str(e)}")
                    continue

            # Join all characters into final formula
            formula = ''.join(decoded_chars)
            
            # Optional: Clean up any remaining special characters or whitespace
            formula = formula.strip()
            
            return formula

        except Exception as e:
            print(f"Error decoding formula: {str(e)}")
            return ""

# Example usage in your dataset class:
def __getitem__(self, idx):
    # Your existing __getitem__ code here
    formula = self.dataset[idx]['formula'].strip()
    image = self.dataset[idx]['image'].convert('L')

    if self.transform:
        image = self.transform(image)

    # Encode formula
    try:
        encoded_formula = torch.tensor([
            self.char_to_idx[char] 
            for char in formula 
            if char in self.char_to_idx
        ])
    except KeyError as e:
        print(f"Warning: Character not in vocabulary: {str(e)}")
        encoded_formula = torch.tensor([])

    return image, encoded_formula
    #end of my changes 
    
    def save_vocab(self, vocab_path='vocab.json'):
        # Save the vocabulary as a JSON file
        vocab_dict = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char
        }
        with open(vocab_path, 'w') as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=4)
        print(f"Vocabulary saved to {vocab_path}")
