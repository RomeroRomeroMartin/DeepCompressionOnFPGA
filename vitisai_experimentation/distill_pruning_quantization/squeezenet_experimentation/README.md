# Distillation + Pruning + Quantization of SqueezeNet
This folder contains the scripts and resources to apply pruning and quantization to a distilled SqueezeNet (distilled from original ResNet-50)

## Folder Structure

### 1. **[.vai](./.vai)**
This hidden folder contains results and information generated automatically during the pruning process. It stores details such as pruning metrics and configurations applied to the models.

### 2. **[models](./models)**
This folder contains the distilled SqueezeNet and the pruned SqueezeNet models that are ready to be quantized. The training of distillations models is available [here](../../../knowledge_distillation/).

### 3. **[quantization_squeezenet](./quantization_squeezenet)**
This folder contains the results of applying quantization to **20% pruned** SqueezeNet. It includes de quantized model, the architecture of quantized model and information files about the quantization process. Inside this folder there is the folder [compiled_model](./quantization_squeezenet/compiled_model/), that have the final model compiled and ready to be executed in the ZCU104.

### 4. **[ptq_squeezenet.py](./ptq_squeezenet.py)**
A Python script to perfom Post-Training Quantization of the distilled and pruned SqueezeNet.

### 5. **[squeezenet_pruning.py](./squeezenet_pruning.py)**
A Python script to perfom a pruning of **20%** to the distilled SqueezeNet.

## How to run the scripts

To execute the Python script for prune and quantize the models, simply run the following command:

```bash
python squeezenet_pruning.py
```
or 
```bash
python ptq_squeezenet.py
```
### Note:

Make sure to update the paths to the models inside the ``squeezenet_pruning.py`` and ``ptq_squeezenet.py`` scripts according to the model you want to prune and quantize. The paths should point to the model located in the ``/models`` folder (or any other directory where your models are stored). In the script  ``squeezenet_pruning.py`` you can also change the type of pruning applied (you can select `one_step` or `iterative`). In the script ``ptq_squeezenet.py`` you can change the path to select which model you want to quantize, and also the name of the directory were the results are saved.

