import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader

from pytorch_nndct import nn as nndct_nn
from pytorch_nndct.nn.modules import functional
from pytorch_nndct import QatProcessor

from common import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device='cpu'
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
#for param in model.parameters():
#    param.requires_grad = False

# Añadimos una capa adicional
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),  # Nueva capa oculta
    nn.ReLU(),                             # Activación
    nn.Dropout(0.5),                       # Dropout para regularización
    nn.Linear(256, 256) ,                    
    nn.Linear(256, 10)                     # Capa final con 10 clases
)

# Solo entrenamos las capas añadidas
for param in model.fc.parameters():
    param.requires_grad = True

# Añadir stubs de cuantización
class QuantizedResNet50(nn.Module):
    def __init__(self, model):
        super(QuantizedResNet50, self).__init__()
        self.quant = nndct_nn.QuantStub()
        self.model = model
        self.dequant = nndct_nn.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)  # Cuantizar entrada
        x = self.model(x)  # Modelo original
        x = self.dequant(x)  # Descuantizar salida
        return x
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
    return model
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
# Load the model
mode='train'
output_dir='qat_results/20epoch_results/'
model = QuantizedResNet50(model)
#model.load_state_dict(torch.load('models/modelo_completo_resnet50_weights.pth'))
model.to(device)
qat_processor = QatProcessor(model, torch.randn([64, 3, 224, 224], dtype=torch.float32, device=device), bitwidth=8)
# Step 1: Get quantized model and train it.
quantized_model = qat_processor.trainable_model(allow_reused_module=True)
criterion = nn.CrossEntropyLoss()
#criterion = criterion.to(device)
print(dir(quantized_model.parameters()))
optimizer = optim.Adam(quantized_model.parameters(), lr=0.001)
#best_ckpt = train(quantized_model, trainloader, testloader, criterion, device_ids=None)
#quantized_model.load_state_dict(torch.load(best_ckpt)['state_dict'])

# Step 2: Train the quantized model
quantized_model=train_model(quantized_model, trainloader, criterion, optimizer, num_epochs=20)
deployable_model = qat_processor.to_deployable(quantized_model, output_dir)
#validate(testloader, deployable_model, criterion, device)
# Step 3: Test the quantized model
test_model(deployable_model, testloader)

# Step 4: Export the quantized model
deployable_model = qat_processor.deployable_model(
        output_dir, used_for_xmodel=True)
val_subset = torch.utils.data.Subset(testset, list(range(1)))
subset_loader = torch.utils.data.DataLoader(
    val_subset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True)
# Must forward deployable model at least 1 iteration with batch_size=1
for images, _ in subset_loader:
    deployable_model(images)
qat_processor.export_xmodel(output_dir)
