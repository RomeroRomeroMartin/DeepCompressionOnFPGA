# Quantization of Distilled Models

This directory contains resources and scripts related to the quantization of the previously distilled models. The distilled models are obtained from the respective folders for MobileNet, ResNet-18, and SqueezeNet.

## Folder Structure

### 1. **[quantization_mobilenet](./quantization_mobilenet)**
This folder contains the results of applying quantization to the **MobileNet** distilled model. It includes the quantized model, its architecture, and associated files related to the quantization process. Inside this folder there is the folder [compiled_model](./quantization_mobilenet/compiled_model/), that have the final model compiled and ready to be executed in the ZCU104.

### 2. **[quantization_resnet18](./quantization_resnet18)**
This folder contains the results of applying quantization to the **ResNet-18** distilled model. It includes the quantized model, its architecture, and associated files related to the quantization process. Inside this folder there is the folder [compiled_model](./quantization_resnet18/compiled_model/), that have the final model compiled and ready to be executed in the ZCU104.

### 3. **[quantization_squeezenet](./quantization_squeezenet)**
This folder contains the results of applying quantization to the **SqueezeNet** distilled model. It includes the quantized model, its architecture, and associated files related to the quantization process. Inside this folder there is the folder [compiled_model](./quantization_squeezenet/compiled_model/), that have the final model compiled and ready to be executed in the ZCU104.

### 4. **[models](./models)**
This folder contains the distilled models (MobileNet, ResNet-18, and SqueezeNet) that are ready to be quantized. These models are obtained from the respective distillation process.

### 5. **[quantize.py](./quantize.py)**
A Python script for performing the quantization of the distilled models. This script takes the distilled models as input, applies the Post-Training Quantization (PTQ) process, and generates quantized versions of the models.

## Usage

1. **Distill the models**: First, use the distillation process to obtain the distilled models (MobileNet, ResNet-18, and SqueezeNet).
2. **Quantize the models**: Run the `[quantize.py](./quantize.py)` script to apply Post-Training Quantization (PTQ) to the distilled models.
3. **Results**: The quantized models will be saved in the respective folders (`quantization_mobilenet`, `quantization_resnet18`, `quantization_squeezenet`).

## How to run the scripts

To execute the Python script for quantizing the distilled models, simply run the following command:

```bash
python quantize.py
```
### Note:

Make sure to update the paths to the models inside the ``quantize.py`` script according to the model you want to quantize. The paths should point to the pruned models located in the ``/models`` folder (or any other directory where your models are stored).
