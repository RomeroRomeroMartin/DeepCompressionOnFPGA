import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet50
from common import *

# Check if GPU is available, and if not, use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Loading the CIFAR-10 dataset:
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

#Dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)


teacher = resnet50(weights=True)

# Congelamos todas las capas menos la fc original
for param in teacher.parameters():
    param.requires_grad = False

# Añadir una capa adicional
teacher.fc = nn.Sequential(
    nn.Linear(teacher.fc.in_features, 256),  # Nueva capa oculta
    nn.ReLU(),                             # Activación
    nn.Dropout(0.5),                       # Dropout para regularización
    nn.Linear(256,256),                    # Nueva capa oculta
    nn.Linear(256, 10)                     # Capa final con 10 clases
)

# Solo entrenamos las capas añadidas
for param in teacher.fc.parameters():
    param.requires_grad = True



# Movemos el modelo a la GPU si está disponible
model = teacher.to(device)

# Define the models architecture
'''class StNN(nn.Module):
    def __init__(self, num_classes=10):
        super(StNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x'''

class StNN(nn.Module):
    def __init__(self, num_classes=10):
        super(StNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1),  # Aumentamos los filtros
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # Nueva capa convolucional
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*7*7, 1024),  # Aumentamos el tamaño de entrada y salida
            nn.ReLU(),
            nn.Dropout(0.3),       # Aumentamos el dropout para evitar sobreajuste
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
'''class StNN(nn.Module):
    def __init__(self):
        super(StNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = (3,3), stride = (1,1), 
            padding = (1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size = (3,3), stride = (1,1), 
            padding = (1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, 
            dilation=1, ceil_mode=False)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = (3,3), stride = (1,1), 
            padding = (1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size = (3,3), stride = (1,1), 
            padding = (1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, 
            dilation=1, ceil_mode=False)
        )
        self.pool1 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 10)
        self.dropout_rate = 0.5
   
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x'''
    
student_noKD = StNN()
student_noKD.to(device)


teacher=torch.load('models/teacher.pth')
teacher.to(device)
acc_teacher= test(teacher, test_loader, device)


total_params_deep = "{:,}".format(sum(p.numel() for p in teacher.parameters()))
print(f"Teacher parameters: {total_params_deep}")
total_params_light = "{:,}".format(sum(p.numel() for p in student_noKD.parameters()))
print(f"Student parameters: {total_params_light}")

student = StNN()
student.to(device)

# Train the student with knowledge distillation
print('TRAINING STUDENT WITH KNOWLEDGE DISTILLATION')
train_knowledge_distillation(teacher=teacher, student=student, train_loader=train_loader, epochs=20, learning_rate=0.001, T=2, 
                             soft_target_loss_weight=0.25, ce_loss_weight=0.75, device=device)
acc_student= test(student, test_loader, device)
torch.save(student, 'models/student4_kl.pth')

# Compare the student test accuracy with and without the teacher, after distillation
print(f"Teacher accuracy: {acc_teacher:.2f}%")
print(f"Student accuracy with CE + KD: {acc_student:.2f}%")


