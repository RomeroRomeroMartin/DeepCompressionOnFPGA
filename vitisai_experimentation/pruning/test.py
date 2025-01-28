import os
import sys
import argparse
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
#from pytorch_nndct.apis import torch_quantizer, dump_xmodel
from torchvision.models import resnet50
#from pytorch_nndct import get_pruning_runner

# Definir la arquitectura del modelo como antes
def base_model():
    # Cargar el modelo base ResNet-50
    model = resnet50(weights=True)

    # Congelar todas las capas menos la fc
    for param in model.parameters():
        param.requires_grad = False

    # Modificar la capa totalmente conectada (fc)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),  # Nueva capa oculta
        nn.ReLU(),                             # Activación
        nn.Dropout(0.5),                       # Dropout
        nn.Linear(256,256),                    # Nueva capa oculta
        nn.Linear(256, 10) 
    )

    return model
def test(model, device, test_loader):
    '''
    test the model
    '''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(test_loader.dataset), acc))

    return
# Transformar las imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Cambiamos el tamaño de las imágenes para que se ajusten a ResNet-50
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizamos las imágenes
])
# Cargar el conjunto de datos de prueba
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

# Definir el modelo
model=base_model()
model.load_state_dict(torch.load('models/cg_pruned_os02_sparse.pth'))
#model.load_state_dict(torch.load('resnet18_sparse.pth'))

model.to('cuda')
# Test the model
test(model, 'cuda', testloader)
