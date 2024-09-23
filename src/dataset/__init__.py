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

    def decode_formula(self, encoded_formula):
        return ''.join(
            [self.idx_to_char[idx.item()] for idx in encoded_formula]
        )
