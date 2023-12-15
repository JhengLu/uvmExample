import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
hidden_size = 50000
num_classes = 1000
num_epochs = 50
batch_size = 500
learning_rate = 0.001
sub_batch_size = 100  # Sub-batch size for processing

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Model
class LargerNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LargerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

model = LargerNN(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Process
for epoch in range(num_epochs):
    pbar = tqdm(train_loader)
    for i, (images, labels) in enumerate(pbar):
        images = images.reshape(-1, 28*28)
        labels = labels

        # Split batch into sub-batches
        sub_batch_losses = []
        for j in range(0, len(images), sub_batch_size):
            sub_images = images[j:j+sub_batch_size].to(device)
            sub_labels = labels[j:j+sub_batch_size].to(device)

            # Forward pass
            outputs = model(sub_images)
            loss = loss_function(outputs, sub_labels)
            sub_batch_losses.append(loss)

            # Backward pass for sub-batch
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Move the data back to CPU to free up GPU memory
            sub_images.cpu()
            sub_labels.cpu()

        avg_loss = sum(sub_batch_losses) / len(sub_batch_losses)
        pbar.set_description("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, avg_loss.item()))

torch.save(model.state_dict(), 'model.ckpt')
