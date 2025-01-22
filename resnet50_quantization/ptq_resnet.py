import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torch.quantization.quantize_fx import prepare_fx, convert_fx


# Comprobamos si hay GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


model=torch.load('ResnetModels/modelo_completo_resnet50_cifar10.pth')
model.to(device)


backend = "qnnpack"
model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend
qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping(backend)

model.eval()
example_inputs = (torch.randn(1, 3, 224, 224),)
model = prepare_fx(model, qconfig_mapping, example_inputs)


with torch.no_grad():
    for images, _ in testloader:
        images = images.to(device)  # Los datos deben estar en la CPU para la calibración
        model(images)

model = convert_fx(model)

traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
torch.jit.save(traced_model, 'ResnetModels/modelo_ptq_resnet50_cifar10_jit.pth')


# Evaluación en GPU
traced_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)  # Mover las imágenes a la GPU
        labels = labels.to(device)  # Mover las etiquetas a la GPU
        images_in = torch.quantization.QuantStub()(images)
        outputs = traced_model(images_in)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test accuracy of the quantized model on GPU: {100 * correct / total:.2f}%')




