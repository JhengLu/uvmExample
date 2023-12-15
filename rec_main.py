import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchrec.modules.embedding_modules import EmbeddingBagCollection, EmbeddingConfig
from torchrec.modules.interaction_modules import MLPInteraction
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

# Create a fake dataset for demonstration
dataset = datasets.FakeData(size=60000, image_size=(1, 28, 28), num_classes=10, transform=transforms.ToTensor())
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
        self.interaction = MLPInteraction(input_dim=embedding_dim*2, layer_sizes=[64, 32, 1])
    
    def forward(self, user_id, item_id):
        user_emb = self.user_embeddings(user_id)
        item_emb = self.item_embeddings(item_id)
        interaction = torch.cat([user_emb, item_emb], dim=1)
        return self.interaction(interaction)

model = RecSysModel(num_users, num_items, embedding_dim).to(device)

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        # Generate random user and item IDs for demonstration
        user_ids = torch.randint(0, num_users, (batch_size,))
        item_ids = torch.randint(0, num_items, (batch_size,))
        targets = torch.rand(batch_size, 1).float()

        user_ids, item_ids, targets = user_ids.to(device), item_ids.to(device), targets.to(device)

        # Forward pass
        outputs = model(user_ids, item_ids).squeeze()
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item()}')

print("Training complete.")
