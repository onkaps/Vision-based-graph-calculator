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
        
        # Get formula tokens and pad them
        formula_tokens = [item['formula_tokens'].squeeze(0) for item in batch]  # Remove extra batch dimension
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

        images, formulas = images.to(device), formulas.to(device)
        # No ground truth captions during testing
        with torch.no_grad():
            output = model(images, None)  # Pass only images to the model

        # Get predicted token IDs
        print(output)
        predicted_token_ids = torch.argmax(output, dim=-1)

        # Convert predicted token IDs to LaTeX string
        predicted_formula = tokenizer.decode(predicted_token_ids.squeeze().tolist(), skip_special_tokens=True)

        print("Predicted LaTeX formula:", predicted_formula)
