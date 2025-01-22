# Quantization the Base Model

This directory contains resources and scripts related to the quantization of the original **ResNet-50**. 

## Folder Structure

### 1. **[.vai_qat](./.vai_qat)**
This hidden folder contains results and information generated automatically during the Quantization Aware Training process. 

### 2. **[models](./models)**
This folder contains the original **ResNet-50** model that are ready to be quantized. 

### 3. **[qat_results](./qat_results)**
This folder contains the results of applying Quantization Aware Training to a **ResNet-50** trained from scratch (without using the weights of the model in `models`folder). It includes the quantized model, its architecture, and associated files related to the quantization process. Inside this folder there is the folder [compiled_model](./qat_results/compiled_model/), that have the final model compiled and ready to be executed in the ZCU104.

### 4. **[qat_results_with_weights](./qat_results_with_weights)**
This folder contains the results of applying Quantization Aware Training to the **ResNet-50** model trained before (the model in `models` folder). It includes the quantized model, its architecture, and associated files related to the quantization process. Inside this folder there is the folder [compiled_model](./qat_results_with_weights/compiled_model/), that have the final model compiled and ready to be executed in the ZCU104.

### 5. **[quantization_output](./quantization_output)**
This folder contains the results of applying Post-Training Quantizatino to the **ResNet-50** model trained before (the model in `models` folder). It includes the quantized model, its architecture, and associated files related to the quantization process. Inside this folder there is the folder [compiled_model](./quantization_output/compiled_model/), that have the final model compiled and ready to be executed in the ZCU104.

### 6. **[common.py](./common.py)**
A Python script with auxilar code.

### 7. **[compile.sh](./compile.sh)**
A script used to compile the quantized models. After quantization, this script generates a compiled model that is optimized and ready for deployment on the ZCU104 FPGA. It handles the compilation process, producing the final model that can be executed on the target hardware.

### 8. **[qat_quant_weights.py](./qat_quant_weights.py)**
A Python script to perform Quantization Aware Training to the trained **ResNet-50** model.

### 9. **[qat_quantization.py](./qat_quantization.py)**
A Python script to perform Quantization Aware Training to a ResNet-50 model from scratch.

### 10. **[quantize.py](./quantize.py)**
A Python script to perform the Post-Training Quantization of the **ResNet-50** model.

### 11. **[test.py](./test.py)**
A Python script to perform an accuracy test of a model.


## How to run the scripts

To execute the Python script for quantizing the models, simply run the following commands depending of what type of quantization you want to execute:

```bash
python qat_quant_weights.py
python qat_quantization.py
python quantize.py
```
### Note:

Check the paths to the models inside the ``quantize.py``, `qat_quant_weights.py`and `qat_quantization.py` script are correct. The paths should point to the orignal ResNet-50 model located in the ``/models`` folder (or any other directory where your models are stored). Also you can change (if you want) the name of the directory where the quantization results are saved.
