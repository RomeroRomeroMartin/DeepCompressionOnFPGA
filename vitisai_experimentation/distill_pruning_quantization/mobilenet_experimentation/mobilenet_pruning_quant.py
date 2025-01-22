import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pytorch_nndct import get_pruning_runner
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from pytorch_nndct.apis import torch_quantizer
import os 

CIFAR10_TRAINSET_SIZE=50000
batch_size=64
ngpus_per_node = torch.cuda.device_count()
T=2
epochs=10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#device='cpu'
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Cambiamos el tamaño de las imágenes para que se ajusten a ResNet-50
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizamos las imágenes
])

full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Dividimos el conjunto de entrenamiento en entrenamiento y validación
train_size = int(0.8 * len(full_trainset))  # Usamos el 80% para entrenamiento
val_size = len(full_trainset) - train_size  # El resto será para validación
trainset, valset = random_split(full_trainset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

# Creamos los loaders para entrenamiento, validación y test
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self, name, fmt=':f'):
    self.name = name
    self.fmt = fmt
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions
    for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res

def eval_fn(model, dataloader):
  top1 = AverageMeter('Acc@1', ':6.2f')
  model.eval()
  with torch.no_grad():
    for i, (images, targets) in enumerate(dataloader):
        images = images.cuda()
        targets = targets.cuda()
        outputs = model(images)
        acc1, _ = accuracy(outputs, targets, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
  return top1.avg

def train_fn(model, trainloader, optimizer, criterion, device):
    """Función para entrenar el modelo."""
    model.train()
    total_loss = 0
    for images, targets in trainloader:
        images, targets = images.to(device), targets.to(device)

        # Adelante
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Atrás
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(trainloader)
def calibration_fn(model, train_loader, number_forward=100):
  model.train()
  print("Adaptive BN atart...")
  with torch.no_grad():
    for index, (images, target) in enumerate(train_loader):
      images = images.cuda()
      model(images)
      if index > number_forward:
        break
  print("Adaptive BN end...")

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

input_signature = torch.randn([1, 3, 224, 224], dtype=torch.float32).to(device)

model = models.mobilenet_v2(weights="IMAGENET1K_V2")
model.classifier = nn.Sequential(
    nn.Dropout(0.5),  # Dropout con probabilidad 0.5
    nn.Linear(model.last_channel, 128),  # Capa completamente conectada de 128 neuronas
    nn.ReLU(),  # Activación ReLU
    nn.Linear(128, 10)  # Capa de salida para 10 clases (CIFAR-10)
)
model.load_state_dict(torch.load('models/mobilenet_distilled2_weights.pth'))
model.to(device)
mode='one_step'
ratio=0.95
runner = get_pruning_runner(model, input_signature, mode)

if mode=='iterative':
  runner.ana(eval_fn, args=(valloader,))

  model = runner.prune(removal_ratio=ratio)
  optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizador con tasa de aprendizaje baja

elif mode=='one_step':
  runner.search(
          calibration_fn=calibration_fn,
          eval_fn=eval_fn,
          num_subnet=20,
          removal_ratio=ratio,
          calib_args=(trainloader,),
          eval_args=(valloader,))

      # index=None: select optimal subnet automatically
  slim_model = runner.prune(
      removal_ratio=ratio, mode='slim', index=None, channel_divisible=2)
  #model = slim_model
  optimizer = optim.Adam(slim_model.parameters(), lr=0.001)  # Optimizador con tasa de aprendizaje baja

epochs=10
criterion = nn.CrossEntropyLoss()  # Función de pérdida
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

print("Iniciando Fine-Tuning del modelo pruneado...")
for epoch in range(epochs):
    train_loss = train_fn(slim_model, trainloader, optimizer, criterion, device)
    val_acc = eval_fn(slim_model, valloader)
    scheduler.step()

    print(f"Época {epoch+1}/{epochs}, Pérdida de entrenamiento: {train_loss:.4f}, Precisión en validación: {val_acc:.2f}%")

# Evaluamos en el conjunto de prueba después del Fine-Tuning
test_acc = eval_fn(slim_model, testloader)
print(f"Accuracy of pruned model: {test_acc:.2f}%")


# QUANTIZATION
quant_model = './quantization_mobilenet095'  # Cambiar a un directorio en tu espacio de trabajo
os.makedirs(quant_model, exist_ok=True)  # Crea el directorio si no existe

quant_mode='calib'

rand_in = torch.randn([1, 3, 224, 224])
quantizer = torch_quantizer(quant_mode, slim_model, (rand_in), output_dir=quant_model) 
quantized_model = quantizer.quant_model

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

print('Test quantized model in calibration mode')
test(quantized_model, 'cuda', testloader)

quantizer.export_quant_config()


quant_mode='test'

rand_in = torch.randn([1, 3, 224, 224])
quantizer = torch_quantizer(quant_mode, slim_model, (rand_in), output_dir=quant_model) 
quantized_model = quantizer.quant_model

print('Test quantized model in test mode')
test(quantized_model, 'cuda', testloader)

quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)
