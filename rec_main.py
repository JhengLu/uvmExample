import torch
from torchrec.datasets.movielens import MovieLens1MDataset
from torchrec.modules.embedding_modules import EmbeddingBagCollection, EmbeddingConfig
from torchrec.modules.interaction_modules import MLPInteraction
from torch.utils.data import DataLoader
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    EmbeddingComputeDevice
)

# Assuming a CUDA device with UVM capabilities
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 1024
learning_rate = 0.001
num_epochs = 10

# Load Dataset (MovieLens 1M for example)
dataset = MovieLens1MDataset("/path/to/movielens/")
train_dataset = dataset.train_dataset()
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the Model
class RecSysModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        embedding_config = EmbeddingConfig(
            compute_kernel=EmbeddingComputeKernel.FUSED_UVM,
            compute_device=EmbeddingComputeDevice.CUDA
        )
        self.user_embeddings = EmbeddingBagCollection(
            num_embeddings=6040,  # number of users in MovieLens 1M
            embedding_dim=64,
            config=embedding_config,
        )
        self.item_embeddings = EmbeddingBagCollection(
            num_embeddings=3706,  # number of movies in MovieLens 1M
            embedding_dim=64,
            config=embedding_config,
        )
        self.interaction = MLPInteraction(input_dim=128, layer_sizes=[64, 32, 1])
    
    def forward(self, user_id, item_id):
        user_emb = self.user_embeddings(user_id)
        item_emb = self.item_embeddings(item_id)
        interaction = torch.cat([user_emb, item_emb], dim=1)
        return self.interaction(interaction)

model = RecSysModel().to(device)

# Loss and Optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    for batch in train_dataloader:
        user_id = batch["user_id"].to(device)
        item_id = batch["item_id"].to(device)
        labels = batch["label"].float().to(device)

        # Forward pass
        outputs = model(user_id, item_id).squeeze()
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

print("Training complete.")
