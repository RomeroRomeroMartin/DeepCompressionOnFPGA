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
from torchvision import models
from torchvision.models import resnet50
from pytorch_nndct.pruning import Pruner


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

# Cargamos el conjunto de datos CIFAR-10
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Cambiamos el tamaño de las imágenes para que se ajusten a ResNet-50
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizamos las imágenes
])

# Cargamos el conjunto de datos de prueba
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

# Cargamos el modelo pre-entrenado
model = models.mobilenet_v2(weights="IMAGENET1K_V2")
model.classifier = nn.Sequential(
    nn.Dropout(0.5),  # Dropout con probabilidad 0.5
    nn.Linear(model.last_channel, 128),  # Capa completamente conectada de 128 neuronas
    nn.ReLU(),  # Activación ReLU
    nn.Linear(128, 10)  # Capa de salida para 10 clases (CIFAR-10)
)
model.to('cuda')
# Cargamos el modelo pruned
pruner = Pruner()
pruned_model = pruner.slim_model(model)
pruned_model.load_state_dict(torch.load('models/cg_pruned_mobilenet02_sparse.pth')['model_state_dict'])
# Evaluamos el modelo pruned
test(pruned_model, 'cuda', testloader)
