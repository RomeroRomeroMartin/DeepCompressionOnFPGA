import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformaciones para el dataset CIFAR-10
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Cargamos el dataset CIFAR-10
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# Cargar el modelo maestro (teacher)
teacher_model = torch.load('models/teacher.pth')  # Asegúrate de que este modelo esté entrenado
teacher_model.to(device)student_model = models.squeezenet1_0(pretrained=True)
student_model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))  # Cambiar para 10 clases
# Optimizer
optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)

# Función de pérdida
def distillation_loss(y_student, y_teacher, temperature):
    loss = nn.KLDivLoss()(F.log_softmax(y_student / temperature, dim=1),
                          F.softmax(y_teacher / temperature, dim=1)) * (temperature ** 2)
    return loss

# Entrenamiento del modelo de distilación
for epoch in range(num_epochs):
    student_model.train()
    total_loss = 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward teacher model para obtener la salida
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)

        # Forward student model
        student_outputs = student_model(inputs)

        # Pérdida en la salida final
        student_loss = nn.CrossEntropyLoss()(student_outputs, labels)
        distillation_loss_value = distillation_loss(student_outputs, teacher_outputs, temperature)

        # Total loss
        total_loss = student_loss + distillation_loss_value

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item():.4f}')

# Guardar el modelo destilado
torch.save(student_model.state_dict(), 'models/squeezenet_distilled.pth')

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

# Evaluar el modelo destilado
test_model(student_model, testloader)
