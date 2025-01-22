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
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

DIVIDER = '-----------------------------------------'
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
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Cambiamos el tamaño de las imágenes para que se ajusten a ResNet-50
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizamos las imágenes
])

quant_model = '/workspace/quant_resnet50/quantization_output/'  # Cambiar a un directorio en tu espacio de trabajo
os.makedirs(quant_model, exist_ok=True)  # Crea el directorio si no existe

quant_mode='test'
# load trained model
model=torch.load('models/modelo_completo_resnet50_cifar10.pth')

rand_in = torch.randn([1, 3, 224, 224])
quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model) 
quantized_model = quantizer.quant_model

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)


test(quantized_model, 'cuda', testloader)

# export config
if quant_mode == 'calib':
    quantizer.export_quant_config()
if quant_mode == 'test':
    quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)



