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
from torchvision.models import resnet50
import torch.nn.utils.prune as prune
from pytorch_nndct import get_pruning_runner
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIVIDER = '-----------------------------------------'

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
# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Cambiamos el tamaño de las imágenes para que se ajusten a ResNet-50
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizamos las imágenes
])

quant_model = './095results/'  # Cambiar a un directorio en tu espacio de trabajo
#os.makedirs(quant_model, exist_ok=True)  # Crea el directorio si no existe



# load trained model
model=base_model()
model.load_state_dict(torch.load('models/cg_pruned_os095_sparse.pth'))
model.to(device)

input_signature = torch.randn([1, 3, 224, 224], dtype=torch.float32).to(device)
# Pruning
mode='one_step'
runner = get_pruning_runner(model, input_signature, mode)
slim_model = runner.prune(removal_ratio=0.95, mode='slim')
slim_model.load_state_dict(torch.load('models/cg_pruned_os095_mask.pth'))
# Test the model
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
test(slim_model, 'cuda', testloader)

# Quantization
quant_mode='calib'

rand_in = torch.randn([1, 3, 224, 224])
quantizer = torch_quantizer(quant_mode, slim_model, (rand_in), output_dir=quant_model) 
quantized_model = quantizer.quant_model
# Test the model
print('Accuracy quantized model in calibration mode')
test(quantized_model, 'cuda', testloader)

quantizer.export_quant_config()

#Quantized model in test mode
quant_mode='test'

rand_in = torch.randn([1, 3, 224, 224])
quantizer = torch_quantizer(quant_mode, slim_model, (rand_in), output_dir=quant_model) 
quantized_model = quantizer.quant_model
# Test the model
print('Accuracy quantized model in test mode')
test(quantized_model, 'cuda', testloader)
# Export the quantized model
quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)

