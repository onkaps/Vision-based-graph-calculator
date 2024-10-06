import torch
import torch.nn as nn


class IM2LatexModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, eos_index=0):
        super(IM2LatexModel, self).__init__()

        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # LSTM for sequence generation
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_layers, batch_first=True)
        self.end_symbol = eos_index
        # Embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)

        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, image, formula=None, teacher_forcing_ratio=0.5):
        # Extract features from image
        features = self.cnn(image)
        features = features.view(features.size(0), -1, 256)  # Reshape for LSTM input

        # Initialize hidden state with image features
        h0 = torch.zeros(self.lstm.num_layers, features.size(0), self.lstm.hidden_size).to(image.device)
        c0 = torch.zeros(self.lstm.num_layers, features.size(0), self.lstm.hidden_size).to(image.device)

        # If we're training, use teacher forcing
        if self.training and formula is not None:
            embedded = self.embed(formula)
            output, _ = self.lstm(embedded, (h0, c0))
            output = self.fc(output)
        else:
            # For inference or if no formula is provided
            output = []
            current_symbol = torch.zeros(image.size(0), 1).long().to(image.device)  # Start symbol
            for _ in range(100):  # Maximum formula length
                embedded = self.embed(current_symbol)
                lstm_out, (h0, c0) = self.lstm(embedded, (h0, c0))
                symbol_prob = self.fc(lstm_out)
                current_symbol = symbol_prob.argmax(2)
                output.append(current_symbol)
                if (current_symbol == self.end_symbol).all():
                    break
            output = torch.cat(output, dim=1)

        return output