import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

# Configuraciones
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
epochs = 10
temperature = 5.0  # Temperatura para la distilación en la salida
alpha = 0.5  # Peso para la pérdida de clasificación vs distilación en la salida
beta = 0.5  # Peso para la distilación entre capas

# Cargar modelos
# Teacher models
resnet50=torch.load('models/teacher.pth')
resnet50.to(device)
resnet50.eval()

# Student model
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Sequential(
    nn.Linear(resnet18.fc.in_features, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
resnet18.to(device)

# Extraer capas intermedias para distilación
class IntermediateFeatureExtractor(nn.Module):
    def __init__(self, model):
        super(IntermediateFeatureExtractor, self).__init__()
        # Incluye las capas iniciales
        self.pre_layers = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self, x):
        outputs = {}
        # Procesar capas iniciales
        x = self.pre_layers(x)
        # Procesar capas intermedias
        x = self.layer1(x)
        outputs['layer1'] = x
        x = self.layer2(x)
        outputs['layer2'] = x
        x = self.layer3(x)
        outputs['layer3'] = x
        return outputs

# Capas seleccionadas
teacher_layers = ['layer1', 'layer2', 'layer3']
student_layers = ['layer1', 'layer2', 'layer3']

teacher_extractor = IntermediateFeatureExtractor(resnet50)
student_extractor = IntermediateFeatureExtractor(resnet18)

# Dataset y DataLoader
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ajustar tamaño para ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Funciones de pérdida
criterion_ce = nn.CrossEntropyLoss()  # Pérdida de clasificación
criterion_kl = nn.KLDivLoss(reduction='batchmean')  # Pérdida de distilación en salida
criterion_mse = nn.MSELoss()  # Pérdida para distilación entre capas

# Optimizador
optimizer = optim.Adam(resnet18.parameters(), lr=0.001)

# Entrenamiento con distilación entre capas
for epoch in range(epochs):
    resnet18.train()
    total_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Salidas del modelo estudiante y profesor
        student_logits = resnet18(inputs)
        with torch.no_grad():
            teacher_logits = resnet50(inputs)

        # Destilación en la salida (logits)
        teacher_probs = nn.functional.softmax(teacher_logits / temperature, dim=1)
        student_probs = nn.functional.log_softmax(student_logits / temperature, dim=1)
        loss_kl = criterion_kl(student_probs, teacher_probs) * (temperature ** 2)

        # Extraer características intermedias
        teacher_features = teacher_extractor(inputs)
        student_features = student_extractor(inputs)        

        # Distilación entre capas
        projection_layers = nn.ModuleDict({
            layer: nn.Conv2d(
                in_channels=student_features[layer].shape[1], 
                out_channels=teacher_features[layer].shape[1], 
                kernel_size=1
            ).to(device)
            for layer in student_layers
        })

        # Aplicar proyección durante la distilación
        loss_intermediate = 0.0
        for t_layer, s_layer in zip(teacher_layers, student_layers):
            student_proj = projection_layers[s_layer](student_features[s_layer])
            loss_intermediate += criterion_mse(student_proj, teacher_features[t_layer])

        # Pérdida de clasificación
        loss_ce = criterion_ce(student_logits, targets)

        # Pérdida total
        loss = alpha * loss_ce + (1 - alpha) * loss_kl + beta * loss_intermediate

        # Actualización de pesos
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

# Evaluación
resnet18.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = resnet18(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')

# Guardar modelo estudiante
torch.save(resnet18.state_dict(), 'models/resnet18_distilled_intermediate.pth')
