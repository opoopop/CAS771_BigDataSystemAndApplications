import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

class CNN_0_A(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN_0_A, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  
        self.bn0 = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  
        self.bn1 = nn.BatchNorm2d(128)
        
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)  
        self.bn3= nn.BatchNorm2d(512)


        self.pool = nn.MaxPool2d(2, 2)  # 2x2 池化
        self.fc1 = nn.Linear(512 * 2 * 2, 256)  
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn0(self.conv0(x))))

        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 64x64 -> 32x32

        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        


        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
class CNN_0_B(nn.Module):
    def __init__(self):
        super(CNN_0_B, self).__init__()
        
        # Bloc 1 : Conv -> ReLU -> BatchNorm -> MaxPool
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bloc 2 : Conv -> ReLU -> BatchNorm -> MaxPool
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Bloc 3 : Conv -> ReLU -> BatchNorm -> MaxPool
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Bloc 4 : Conv -> ReLU -> BatchNorm -> MaxPool
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 4 * 4, 256)  # Adapté pour TinyImageNet (images de 64x64)
        #self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(256, 5)  # 200 classes pour TinyImageNet
        
        # Dropout pour éviter l'overfitting
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        # Bloc 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Bloc 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Bloc 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Bloc 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(-1, 512 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout appliqué après la première couche fully-connected
        #x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
class MetaModel(nn.Module):
    def __init__(self, input_dim=15, hidden_dim1=128, hidden_dim2=64, num_classes=15):
        super(MetaModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

