import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import os

# Define your custom dataset
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []

        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            for image_name in os.listdir(cls_dir):
                image_path = os.path.join(cls_dir, image_name)
                self.image_paths.append(image_path)
                self.labels.append(self.class_to_idx[cls_name])

        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transformations to be applied to your data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Define hyperparameters
batch_size = 16
lr = 0.01
num_epochs = 50

# Initialize your dataset
dataset = CustomDataset(data_dir='./dataset_bitirme', transform=transform)

print(dataset.classes)

print(len(dataset.classes))
# Split dataset into train, validation, and test sets
train_size = int(0.8 * len(dataset))  
val_size = int(0.1 * len(dataset))  
test_size = len(dataset) - train_size - val_size  

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

# Initialize data loaders for each split
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Load pre-trained MobileNetV3 model
model = mobilenet_v3_small(weights=True)
num_ftrs = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_ftrs, len(dataset.classes))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# Define the learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_set)
    print(f'Training - Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    # Update the learning rate
    scheduler.step()

    # Validation loop
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_epoch_loss = val_loss / len(val_set)
    val_accuracy = 100 * correct / total
    print(f'Validation - Epoch [{epoch+1}/{num_epochs}], Loss: {val_epoch_loss:.4f}, Accuracy: {val_accuracy:.2f}%')

print('Finished Training')

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
print('Model saved as trained_model.pth')
