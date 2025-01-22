import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader

# Comprobamos si hay GPU disponible
device = "cpu"

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

# Cargar el modelo previamente entrenado
#model = torch.load('ResnetModels/modelo_completo_resnet50_cifar10.pth', weights_only=False)


class QuantizableResNet50(nn.Module):
    def __init__(self):
        super(QuantizableResNet50, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()	
        self.model = torch.load('ResnetModels/modelo_completo_resnet50_cifar10.pth', weights_only=False)

    def forward(self, x):
        x=self.quant(x)
        x = self.model(x)
        x=self.dequant(x)
        return x

model=QuantizableResNet50()
model.to(device)


def fuse_resnet50(model):
    model_fused = torch.quantization.fuse_modules(model.model, [
        ['conv1', 'bn1'],
        ['layer1.0.conv1', 'layer1.0.bn1'],
        ['layer1.0.conv2', 'layer1.0.bn2'],
        ['layer1.1.conv1', 'layer1.1.bn1'],
        ['layer1.1.conv2', 'layer1.1.bn2'],
        ['layer1.2.conv1', 'layer1.2.bn1'],
        ['layer1.2.conv2', 'layer1.2.bn2'],
        ['layer2.0.conv1', 'layer2.0.bn1'],
        ['layer2.0.conv2', 'layer2.0.bn2'],
        ['layer2.1.conv1', 'layer2.1.bn1'],
        ['layer2.1.conv2', 'layer2.1.bn2'],
        ['layer2.2.conv1', 'layer2.2.bn1'],
        ['layer2.2.conv2', 'layer2.2.bn2'],
        ['layer3.0.conv1', 'layer3.0.bn1'],
        ['layer3.0.conv2', 'layer3.0.bn2'],
        ['layer3.1.conv1', 'layer3.1.bn1'],
        ['layer3.1.conv2', 'layer3.1.bn2'],
        ['layer3.2.conv1', 'layer3.2.bn1'],
        ['layer3.2.conv2', 'layer3.2.bn2'],
        ['layer4.0.conv1', 'layer4.0.bn1'],
        ['layer4.0.conv2', 'layer4.0.bn2'],
        ['layer4.1.conv1', 'layer4.1.bn1'],
        ['layer4.1.conv2', 'layer4.1.bn2'],
        ['layer4.2.conv1', 'layer4.2.bn1'],
        ['layer4.2.conv2', 'layer4.2.bn2']
    ])
    return model_fused

model = fuse_resnet50(model)

# Configurar el backend de cuantización para GPU
torch.backends.quantized.engine = 'fbgemm'

# Preparar el modelo para la cuantización
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# Calibración: Ejecutar una pasada de datos de entrenamiento (o validación)
model.eval()
with torch.no_grad():
    for images, _ in trainloader:
        images = images.to('cpu')  # Los datos deben estar en la CPU para la calibración
        model(images)

model.to('cpu')   
# Convertir el modelo a su versión cuantizada
model_int8 = torch.quantization.convert(model, inplace=True)


torch.save(model_int8, 'ResnetModels/model_ptq_int8.pth')

# Mover el modelo cuantizado a la GPU
model_int8.to('cpu')

# Evaluación en GPU
model_int8.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images = images.to('cpu')  # Mover las imágenes a la GPU
        labels = labels.to('cpu')  # Mover las etiquetas a la GPU
        images_in = torch.quantization.QuantStub()(images)
        outputs = model_int8(images_in)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test accuracy of the quantized model on GPU: {100 * correct / total:.2f}%')
