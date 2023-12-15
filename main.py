import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import torch.cuda.amp as amp  # For mixed precision training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Your Hyperparameters
input_size = 784
hidden_size = 50000
num_classes = 1000
num_epochs = 50
batch_size = 500
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

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

# Automatic Mixed Precision
scaler = amp.GradScaler()

# Training Process
total_step = len(train_loader)
for epoch in range(num_epochs):
    pbar = tqdm(train_loader)
    for i, (images, labels) in enumerate(pbar):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()

        # Forward pass
        with amp.autocast():  # Mixed precision
            outputs = model(images)
            loss = loss_function(outputs, labels)
        
        # Backward and optimize
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        pbar.set_description("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))

torch.save(model.state_dict(), 'model.ckpt')
