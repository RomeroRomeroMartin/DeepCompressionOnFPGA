import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Configuración del dispositivo (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformaciones para el dataset CIFAR-10
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Tamaño ajustado para ResNet-50
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalización
])

# Cargar el conjunto de prueba CIFAR-10
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# Función para evaluar el modelo
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

# Cargar el modelo deployable
model_path = "deployable.pth"  # Ruta al modelo
model = torch.load(model_path, map_location=device)
deployable_model=model.deployable_model()
# Test del modelo
#deployable_model.to(device)
test_model(deployable_model, testloader)
