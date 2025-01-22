import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
import torch.quantization as quantization
from torch.ao.quantization import QuantStub, DeQuantStub

# Comprobamos si hay GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cpu'
# Transformaciones para el dataset CIFAR-10
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Cambiamos el tamaño de las imágenes para que se ajusten a ResNet-50
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizamos las imágenes
])

# Cargamos el dataset CIFAR-10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# Definimos el modelo ResNet-50 preentrenado
model = resnet50(weights=True)

# Congelamos todas las capas menos la fc original
for param in model.parameters():
    param.requires_grad = False

# Añadimos una capa adicional
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),  # Nueva capa oculta
    nn.ReLU(),                             # Activación
    nn.Dropout(0.5),                       # Dropout para regularización
    nn.Linear(256, 10)                     # Capa final con 10 clases
)

# Solo entrenamos las capas añadidas
for param in model.fc.parameters():
    param.requires_grad = True

# Añadir stubs de cuantización
class QuantizedResNet50(nn.Module):
    def __init__(self, model):
        super(QuantizedResNet50, self).__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)  # Cuantizar entrada
        x = self.model(x)  # Modelo original
        x = self.dequant(x)  # Descuantizar salida
        return x

# Función de evaluación
def test_model(model, testloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Precisión en el conjunto de prueba: {100 * correct / total:.2f}%")

#model=torch.load('ResnetModels/modelo_qat_resnet50_cifar10.pth',map_location=torch.device('cpu'))  
model = torch.jit.load('ResnetModels/modelo_qat5bits_resnet50_cifar10_jit.pth')
#model.to(device)
    
model.to('cpu')
test_model(model, testloader)


