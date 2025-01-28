import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pytorch_nndct import get_pruning_runner
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
import torch.nn.utils.prune as prune
CIFAR10_TRAINSET_SIZE=50000
batch_size=64
ngpus_per_node = torch.cuda.device_count()
T=2
epochs=10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#device='cpu'
# Transformaciones para las imágenes
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

input_signature = torch.randn([1, 3, 224, 224], dtype=torch.float32).to(device)

# Load the model
model = models.squeezenet1_0(pretrained=True)
model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))

model.load_state_dict(torch.load('models/squeezenet_distilled.pth'))
model.to(device)

# Pruning mode and ratio
mode='iterative'
ratio=0.2

# Pruning
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

      # index=None: select optimal subnet automaticallyimport torch.nn.utils.prune as prune
  optimizer = optim.Adam(slim_model.parameters(), lr=0.001)  # Optimizador con tasa de aprendizaje baja

# Fine-Tuning
epochs=10
criterion = nn.CrossEntropyLoss()  # Función de pérdida
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

print("Iniciando Fine-Tuning del modelo pruneado...")
for epoch in range(epochs):
    train_loss = train_fn(model, trainloader, optimizer, criterion, device)
    val_acc = eval_fn(model, valloader)
    scheduler.step()

    print(f"Época {epoch+1}/{epochs}, Pérdida de entrenamiento: {train_loss:.4f}, Precisión en validación: {val_acc:.2f}%")

# Evaluamos en el conjunto de prueba después del Fine-Tuning
test_acc = eval_fn(model, testloader)
print(f"Precisión final en el conjunto de prueba: {test_acc:.2f}%")
print(dir(model))
# Guardamos el modelo ajustado. Si ejecutamos en modo iterative guardar como model.state_dict() el modelo completo y como 
# slim_state_dict() el modelo pruneado, si ejecutamos one_step guardar como model.sparse_state_dict() el modelo completo y con state_dict() el modelo pruneado
torch.save(model.state_dict(), 'models/cg_pruned_squeezenet02_sparse.pth')

torch.save(model.slim_state_dict(), 'models/cg_pruned_squeezenet02_mask.pth')
