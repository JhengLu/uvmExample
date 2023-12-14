import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms 
import torch.optim as optim

# Assuming you use Device with UVM capabilities.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Declare Hyperparameters
input_size = 78400
hidden_size = 50000
num_classes = 1000
num_epochs = 5
batch_size = 500
learning_rate = 0.001

# Download MNIST dataset.
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=False)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# Create Data loader.
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Make a Basic Feedforward Model.
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
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = loss_function(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
