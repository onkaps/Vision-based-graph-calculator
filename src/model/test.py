from ..dataset import IM2LatexDataset
from ..utils import BATCH_SIZE
import torch
from torchvision import transforms
from .load import load_model
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence

def test_model():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    test_dataset = IM2LatexDataset(
        split='test',
        transform=transform
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Testing on {device}.')

    model, tokenizer = load_model()

    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        formula_tokens = [item['formula_tokens'].squeeze(0) for item in batch]
        formula_tokens_padded = pad_sequence(formula_tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
        return images, formula_tokens_padded

    test_loader = DataLoader(
        Subset(test_dataset, range(BATCH_SIZE)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    model.to(device)
    model.eval()
    
    for images, formulas in test_loader:
        images = images.to(device)
        
        with torch.no_grad():
            output = model(images)  # Pass only images to the model
        
        # Process each sequence in the batch
        for i in range(output.size(0)):
            sequence = output[i]
            
            # Find the index of the end symbol
            end_index = (sequence == model.end_symbol).nonzero(as_tuple=True)[0]
            
            if len(end_index) > 0:
                # If end symbol is found, truncate the sequence
                predicted_tokens = sequence[:end_index[0]]
            else:
                # If no end symbol, use the whole sequence
                predicted_tokens = sequence
            
            # Convert predicted token IDs to LaTeX string
            predicted_formula = tokenizer.decode(predicted_tokens.tolist(), skip_special_tokens=True)
            
            print(f"Sample {i+1}:")
            print("Predicted LaTeX formula:", predicted_formula)
            print("Ground truth formula:", tokenizer.decode(formulas[i].tolist(), skip_special_tokens=True))
            print()

        break  # Remove this if you want to process all batches