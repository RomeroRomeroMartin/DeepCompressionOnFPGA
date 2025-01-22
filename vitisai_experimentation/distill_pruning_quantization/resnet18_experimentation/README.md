# Distillation + Pruning + Quantization of ResNet-18 
This folder contains the scripts and resources to apply pruning and quantization to a distilled ResNet-18 (distilled from original ResNet-50)

## Folder Structure

### 1. **[.vai](./.vai)**
This hidden folder contains results and information generated automatically during the pruning process. It stores details such as pruning metrics and configurations applied to the models.

### 2. **[models](./models)**
This folder contains the distilled ResNet-18 and the pruned ResNet-18 models that are ready to be quantized.

### 3. **[quantization_resnet18_02](./quantization_resnet18_02)**
This folder contains the results of applying quantization to **20% pruned** ResNet-18. It includes de quantized model, the architecture of quantized model and information files about the quantization process. Inside this folder there is the folder [compiled_model](./quantization_resnet18_02/compiled_model/), that have the final model compiled and ready to be executed in the ZCU104.

### 4. **[ptq_resnet18.py](./ptq_resnet18.py)**
A Python script to perfom Post-Training Quantization of the distilled and pruned ResNet-18.

### 5. **[resnet18_pruning.py](./resnet18_pruning.py)**
A Python script to perfom a pruning of **20%** and to the distilled ResNet-18.

## How to run the scripts

To execute the Python script for prune and quantize the models, simply run the following command:

```bash
python resnet18_pruning.py
```
or 
```bash
python ptq_resnet18.py
```
### Note:

Make sure to update the paths to the models inside the ``resnet18_pruning.py`` and ``ptq_resnet18.py`` scripts according to the model you want to prune and quantize. The paths should point to the model located in the ``/models`` folder (or any other directory where your models are stored). In the script  ``resnet18_pruning.py`` you can also change the type of pruning applied (you can select `one_step` or `iterative`). In the script ``ptq_resnet18.py`` you can change the path to select which model you want to quantize, and also the name of the directory were the results are saved.

