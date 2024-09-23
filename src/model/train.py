import torch
import os
from ..utils import STATE_DICT, MODEL_INFO, BUILD_DIR
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from ..dataset import IM2LatexDataset
from . import IM2LatexModel
import torch.nn.functional as F
from tqdm import tqdm


def train_model():
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001
    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2

# Prepare dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = IM2LatexDataset(
        split='train',
        transform=transform
    )

# Collate function for padding

    def collate_fn(batch):
        images, encoded_formulas = zip(*batch)

        # Pad sequences to the same length
        lengths = [len(seq) for seq in encoded_formulas]
        max_length = max(lengths)
        padded_formulas = [F.pad(seq, (0, max_length - len(seq)), value=0)
                           for seq in encoded_formulas]

        # Stack images and padded formulas
        images = torch.stack(images)
        padded_formulas = torch.stack(padded_formulas)

        return images, padded_formulas

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Initialize model
    model = IM2LatexModel(
        len(train_dataset.vocab),
        EMBED_SIZE,
        HIDDEN_SIZE,
        NUM_LAYERS
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device} for {EPOCHS} epochs.')
    model.to(device)

    progress_bar = tqdm(range(EPOCHS), total=EPOCHS, desc='Training')
    average_loss = 0
    for epoch in progress_bar:
        model.train()
        total_loss = 0
        for images, formulas in train_loader:
            images, formulas = images.to(device), formulas.to(device)
            optimizer.zero_grad()
            outputs = model(images, formulas)
            loss = criterion(
                outputs.view(-1, len(train_dataset.vocab)), formulas.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            break

        progress_bar.set_description(
            'Loss: {0:.6f}'.format(total_loss/len(train_loader))
        )
        average_loss += total_loss

    print('Average Loss: {0:.6f}'.format(average_loss / EPOCHS))

    model_info = {
        'vocab_size': len(train_dataset.vocab),
        'embed_size': EMBED_SIZE,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS
    }

# Save the model
    torch.save(model.state_dict(), os.path.join(BUILD_DIR, STATE_DICT))
    torch.save(model_info, os.path.join(BUILD_DIR, MODEL_INFO))
