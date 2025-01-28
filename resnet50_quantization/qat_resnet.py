import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
import torch.quantization as quantization
from torch.ao.quantization import QuantStub, DeQuantStub
from torch.quantization.quantize_fx import prepare_fx, convert_fx

# Comprobamos si hay GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        #self.quant = QuantStub()
        self.model = model
        #self.dequant = DeQuantStub()

    def forward(self, x):
        #x = self.quant(x)  # Cuantizar entrada
        x = self.model(x)  # Modelo original
        #x = self.dequant(x)  # Descuantizar salida
        return x

# Convertir el modelo para la cuantización
model = QuantizedResNet50(model)

# Mover el modelo a la GPU si está disponible
model = model.to(device)

# Preparar el modelo para Quantization Aware Training (QAT)
model.eval()
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

#model = torch.quantization.prepare(model)
qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping('fbgemm')
example_inputs = (torch.randn(1, 3, 224, 224),)
model = prepare_fx(model, qconfig_mapping, example_inputs)
model.train()


# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Función de entrenamiento
def train_model(model, trainloader, criterion, optimizer, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero grad para el optimizador
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader)}")

    print("Entrenamiento completado.")

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

# Entrenar el modelo con QAT
train_model(model, trainloader, criterion, optimizer, num_epochs=1)

# Convertir el modelo a la versión cuantizada final
model.to('cpu')
model.cpu()
#model = torch.quantization.convert(model.eval(), inplace=True)
model = convert_fx(model.eval())

# Guardar el modelo cuantizado
# Convertir el modelo a TorchScript
traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
torch.jit.save(traced_model, 'ResnetModels/modelo_qat_resnet50_cifar10_jit.pth')

#torch.save(model, 'ResnetModels/modelo_qat_resnet50_cifar10.pth')

# Evaluar el modelo cuantizado
test_model(model, testloader)
