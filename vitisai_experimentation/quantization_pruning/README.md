# Quantization and Pruning

This directory contains resources and scripts related to the quantization of the previous pruned models. The pruned models are obtained in [pruning](../pruning) folder.

## Folder Structure

### 1. **[.vai](./.vai)**
This hidden folder contains results and information generated automatically during the pruning process. It stores details such as pruning metrics and configurations applied to the models.

### 2. **[02results](./02results)**
This folder contains the results of applying quantization to **20% pruning** models. It includes de quantized model, the architecture of quantized model and information files about the quantization process. Inside this folder there is the folder [compiled_model](./02results/compiled_model/), that have the final model compiled and ready to be executed in the ZCU104.

### 3. **[095results](./095results)**
This folder contains the results of applying quantization to **95% pruning** models. It includes de quantized model, the architecture of quantized model and information files about the quantization process. Inside this folder there is the folder [compiled_model](./095results/compiled_model/), that have the final model compiled and ready to be executed in the ZCU104.

### 4. **[models](./models)**
This folder contains the pruned models that are ready to be quantized. This models are obtained in [pruning](../pruning) folder.

### 5. **[compile.sh](./compile.sh)**
A script used to compile the quantized models. After quantization, this script generates a compiled model that is optimized and ready for deployment on the ZCU104 FPGA. It handles the compilation process, producing the final model that can be executed on the target hardware.

### 6. **[quantize.py](./quantize.py)**
A Python script for performing the quantization of pruned models. This script takes the pruned models as input, applies the quantization process (Post-Training Quantization), and generates quantized versions of the models.

## Usage

1. **Prune the models**: First, use the pruning scripts present in [pruning](../pruning) folder to apply the desired pruning level to the models (e.g., 20% or 95% pruning).
2. **Quantize the models**: Run the `[quantize.py](./quantize.py)` script to apply Post-Training Quantization (PTQ) to the pruned models.
3. **Compile the quantized models**: Once the models are quantized, use the `[compile.sh](./compile.sh)` script to compile them for deployment on the ZCU104 FPGA.

## How to run the scripts

To execute the Python script for quantizing the models, simply run the following command:

```bash
python quantize.py
```
### Note:

Make sure to update the paths to the models inside the ``quantize.py`` script according to the model you want to quantize. The paths should point to the pruned models located in the ``/models`` folder (or any other directory where your models are stored).


