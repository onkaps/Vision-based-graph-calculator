import torch
import os
from ..utils import STATE_DICT, MODEL_INFO, BUILD_DIR, BATCH_SIZE
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from ..dataset import IM2LatexDataset
from . import IM2LatexModel
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


def train_model():
    # Hyperparameters
    EPOCHS = 10
    LEARNING_RATE = 0.001
    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2

# Prepare dataset
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    train_dataset = IM2LatexDataset(
        split='train',
        transform=transform
    )

# Collate function for padding

    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        
        # Get formula tokens and pad them
        formula_tokens = [item['formula_tokens'].squeeze(0) for item in batch]  # Remove extra batch dimension
        formula_tokens_padded = pad_sequence(formula_tokens, batch_first=True, padding_value=train_dataset.tokenizer.pad_token_id)
        
        return images, formula_tokens_padded

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Initialize model
    model = IM2LatexModel(
        len(train_dataset.tokenizer),
        EMBED_SIZE,
        HIDDEN_SIZE,
        NUM_LAYERS,
        eos_index = train_dataset.tokenizer.eos_token_id
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
                outputs.view(-1, len(train_dataset.tokenizer)), formulas.view(-1))
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
        'embed_size': EMBED_SIZE,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS
    }

    # Save the model
    torch.save(model.state_dict(), os.path.join(BUILD_DIR, STATE_DICT))
    torch.save(model_info, os.path.join(BUILD_DIR, MODEL_INFO))
    train_dataset.tokenizer.save_pretrained(os.path.join(BUILD_DIR, 'tokenizer'))
