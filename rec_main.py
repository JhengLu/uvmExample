import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchrec.modules.embedding_modules import EmbeddingBagCollection, EmbeddingConfig
from torchrec.distributed.embedding_types import EmbeddingComputeKernel, EmbeddingComputeDevice

# Assuming a CUDA device with UVM capabilities
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_users = 1000  # Simulated number of users
num_items = 1000  # Simulated number of items
batch_size = 1024
learning_rate = 0.001
num_epochs = 10
embedding_dim = 64

# Simulate a dataset
class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        user_id = torch.randint(0, num_users, (1,))
        item_id = torch.randint(0, num_items, (1,))
        label = torch.rand(1).float()
        return user_id, item_id, label

dataset = FakeDataset(60000)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the Model
class RecSysModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        embedding_config = EmbeddingConfig(
            compute_kernel=EmbeddingComputeKernel.FUSED_UVM,
            compute_device=EmbeddingComputeDevice.CUDA
        )
        self.user_embeddings = EmbeddingBagCollection(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
            config=embedding_config,
        )
        self.item_embeddings = EmbeddingBagCollection(
            num_embeddings=num_items,
            embedding_dim=embedding_dim,
            config=embedding_config,
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, user_id, item_id):
        user_emb = self.user_embeddings(user_id)
        item_emb = self.item_embeddings(item_id)
        interaction = torch.cat([user_emb, item_emb], dim=1)
        return self.fc_layers(interaction)

model = RecSysModel(num_users, num_items, embedding_dim).to(device)

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    for i, (user_ids, item_ids, labels) in enumerate(train_dataloader):
        user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)

        # Forward pass
        outputs = model(user_ids, item_ids).squeeze()
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item()}')

print("Training complete.")
