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
from pytorch_nndct.apis import torch_quantizer
import torchvision.models as tmodels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

quant_model = './quantization_resnet18_095'  # Cambiar a un directorio en tu espacio de trabajo
os.makedirs(quant_model, exist_ok=True)  # Crea el directorio si no existe

model = tmodels.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),  # Capa de 256 neuronas
    nn.ReLU(),  # Activación ReLU
    nn.Linear(256, 10)  # Capa de salida para 10 clases (CIFAR-10)
)

model.load_state_dict(torch.load('models/cg_pruned_resnet095_sparse.pth'))


quant_mode='calib'

rand_in = torch.randn([1, 3, 224, 224])
quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model) 
quantized_model = quantizer.quant_model

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

print('Accuracy quantized model in calibration mode')
test(quantized_model, 'cuda', testloader)
quantizer.export_quant_config()



quant_mode='test'

rand_in = torch.randn([1, 3, 224, 224])
quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model) 
quantized_model = quantizer.quant_model

print('Accuracy quantized model in test mode')
test(quantized_model, 'cuda', testloader)

quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)



