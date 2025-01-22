import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os

# Directorio donde se guardarán las imágenes
output_dir = "cifar10_images"

# Crear la carpeta si no existe
os.makedirs(output_dir, exist_ok=True)

# Descargar el dataset CIFAR-10
transform = transforms.Compose([transforms.ToTensor()])
cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Guardar cada imagen en disco
for i, (img, label) in enumerate(cifar10):
    img = transforms.ToPILImage()(img)  # Convertir el tensor a una imagen PIL
    class_name = cifar10.classes[label]
    class_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    img.save(os.path.join(class_dir, f"img_{i}.png"))

print("Imágenes guardadas en el directorio:", output_dir)
