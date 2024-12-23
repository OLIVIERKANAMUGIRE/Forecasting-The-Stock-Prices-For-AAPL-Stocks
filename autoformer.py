import torch
import torch.nn as nn

class DecompositionBlock(nn.Module):
    def __init__(self, kernel_size=25):
        super(DecompositionBlock, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        trend = self.moving_avg(x.permute(0, 2, 1)).permute(0, 2, 1)  # Trend component
        seasonal = x - trend  # Seasonal component
        return seasonal, trend


class Autoformer(nn.Module):
    def __init__(self, input_dim, seq_length, d_model, n_heads, num_layers, kernel_size=25):
        super(Autoformer, self).__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.decoder = nn.Linear(d_model, 1)

        # Decomposition layer
        self.decomposition = DecompositionBlock(kernel_size=kernel_size)

        # Transformer Encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=128),
            num_layers=num_layers,
        )

    def forward(self, x):
        # Decomposition
        seasonal, trend = self.decomposition(x)

        # Combine seasonal and trend for modeling
        x_encoded = self.encoder(seasonal + trend)

        # Transformer-based feature extraction
        transformer_out = self.transformer_encoder(x_encoded.permute(1, 0, 2))  # (seq_len, batch, d_model)

        # Decode the last step
        output = self.decoder(transformer_out[-1])  # Forecast only the last time step
        return output



# Model parameters
input_dim = 1
seq_length = 30
d_model = 64
n_heads = 4
num_layers = 2
kernel_size = 25

# Initialize Autoformer
autoformer_model = Autoformer(input_dim, seq_length, d_model, n_heads, num_layers, kernel_size)
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to device
autoformer_model = autoformer_model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoformer_model.parameters(), lr=0.001)


# Training loop
# Store losses
train_losses = []
val_losses = []

num_epochs = 50
for epoch in range(num_epochs):
    autoformer_model.train()
    optimizer.zero_grad()

    # Convert X_train and y_train to PyTorch tensors and move to device
    # Ensure X_train and y_train are on the same device as the model
    X_train_gpu = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_gpu = torch.tensor(y_train, dtype=torch.float32, device=device)

    # Forward pass
    output = autoformer_model(X_train_gpu)
    loss = criterion(output.squeeze(), y_train_gpu)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # Evaluate the model
    autoformer_model.eval()
    with torch.no_grad():
        # Convert X_test to a PyTorch tensor
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
        predictions = autoformer_model(X_val_tensor)

        # Convert y_test to a PyTorch tensor if it's not already
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device) if not isinstance(y_test, torch.Tensor) else y_test

        test_loss = criterion(predictions.squeeze(), y_val_tensor.squeeze())
        val_losses.append(test_loss.item())
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")