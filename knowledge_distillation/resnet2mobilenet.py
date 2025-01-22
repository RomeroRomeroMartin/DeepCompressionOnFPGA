import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

# Configuraciones
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
epochs = 10
temperature = 5.0  # Temperatura para la distilación
alpha = 0.5  # Peso entre la pérdida de distilación y la pérdida de clasificación

# Cargar modelos
resnet50=torch.load('models/teacher.pth')
resnet50.to(device)
resnet50.eval()

mobilenet = models.mobilenet_v2(weights="IMAGENET1K_V2")
mobilenet.classifier = nn.Sequential(
    nn.Dropout(0.5),  # Dropout con probabilidad 0.5
    nn.Linear(mobilenet.last_channel, 128),  # Capa completamente conectada de 128 neuronas
    nn.ReLU(),  # Activación ReLU
    nn.Linear(128, 10)  # Capa de salida para 10 clases (CIFAR-10)
)
mobilenet.to(device)

# Dataset y DataLoader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Funciones de pérdida
criterion_ce = nn.CrossEntropyLoss()  # Pérdida de clasificación
criterion_kl = nn.KLDivLoss(reduction='batchmean')  # Pérdida de distilación

# Optimizador
optimizer = optim.Adam(mobilenet.parameters(), lr=0.001)

# Entrenamiento con distilación
for epoch in range(epochs):
    mobilenet.train()
    total_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Salidas del modelo estudiante y profesor
        student_logits = mobilenet(inputs)
        with torch.no_grad():
            teacher_logits = resnet50(inputs)
        
        # Salidas suavizadas con temperatura
        teacher_probs = nn.functional.softmax(teacher_logits / temperature, dim=1)
        student_probs = nn.functional.log_softmax(student_logits / temperature, dim=1)
        
        # Pérdida de distilación y clasificación
        loss_kl = criterion_kl(student_probs, teacher_probs) * (temperature ** 2)
        loss_ce = criterion_ce(student_logits, targets)
        loss = alpha * loss_ce + (1 - alpha) * loss_kl

        # Actualización de pesos
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

# Evaluación
mobilenet.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = mobilenet(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')

# Guardar modelo estudiante
torch.save(mobilenet.state_dict(), 'models/mobilenet_distilled2_weights.pth')
