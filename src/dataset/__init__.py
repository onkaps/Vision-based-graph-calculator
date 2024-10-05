import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

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
        self.transform = transform
        self.length = 0

        # Create vocabulary
        self.tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        formula = example['formula']
        tokens = self.tokenizer(formula.strip(), return_tensors='pt', padding = True).input_ids
        return dict(
            image =  self.transform(example['image']) if self.transform else example['image'],
            formula_tokens = tokens
        )

    def __setitem__(self, idx, value):
        self.dataset[idx] = value

