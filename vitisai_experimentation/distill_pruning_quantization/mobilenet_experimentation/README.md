# Distillation + Pruning + Quantization of MobileNet 
This folder contains the scripts and resources to apply pruning and quantization to a distilled MobileNet (distilled from original ResNet-50)

## Folder Structure

### 1. **[.vai](./.vai)**
This hidden folder contains results and information generated automatically during the pruning process. It stores details such as pruning metrics and configurations applied to the models.

### 2. **[models](./models)**
This folder contains the distilled MobileNet and the pruned MobileNet models that are ready to be quantized.

### 3. **[quantization_mobilenet02](./quantization_mobilenet02)**
This folder contains the results of applying quantization to **20% pruned** MobileNet. It includes de quantized model, the architecture of quantized model and information files about the quantization process. Inside this folder there is the folder [compiled_model](./quantization_mobilenet02/compiled_model/), that have the final model compiled and ready to be executed in the ZCU104.

### 4. **[mobilenet_pruning_quant.py](./mobilenet_pruning_quant.py)**
A Python script to perfom a pruning of **20%** and Post-Training Quantization of the distilled MobileNet.

### 5. **[test.py](./test.py)**
A Python script to perform an accuracy test of a model.


## How to run the scripts

To execute the Python script for prune and quantize the models, simply run the following command:

```bash
python mobilenet_pruning_quant.py
```
### Note:

Make sure to update the paths to the models inside the ``mobilenet_pruning_quant.py`` script according to the model you want to prune and quantize. The paths should point to the model located in the ``/models`` folder (or any other directory where your models are stored). In the script you can also change the type of pruning applied (you can select `one_step` or `iterative`)
