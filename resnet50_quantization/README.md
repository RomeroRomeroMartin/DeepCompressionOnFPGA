# Training and Quantization of ResNet-50 with Pytorch

This directory contains resources and scripts related to the training and quantization of a ResNet-18.

## Folder Structure

### 1. **[ResnetModels](./ResnetModels)**
This folder contains the original **ResNet-50** and all the quantized models. 

### 2. **[eval_quantization](./eval_quantization.py)**
A Python script to perform a test to quantized models.

### 3. **[get_model_info.ipynb](./get_model_info.ipynb)**
A Jupyter Notebook to generate the figures with information about the quantized models.

### 4. **[ptq_resnet.py](./ptq_resnet.py)**
A Python script to perform Post-Training Quantization (8 bits) quantization to ResNet-50.

### 5. **[ptq5bits_resnet.py](./ptq5bits_resnet.py)**
A Python script to perform Post-Training Quantization (5 bits) quantization to ResNet-50.

### 6. **[qat_resnet.py](./qat_resnet.py)**
A Python script to perform Quantization Aware Training (8 bits) quantization to ResNet-50.

### 7. **[qat5_resnet.py](./qat5_resnet.py)**
A Python script to perform Quantization Aware Training (5 bits) quantization to ResNet-50.

### 8. **[train_resnet50.py](./train_resnet50.py)**
A Python script to train the original ResNet-50.

## How to run the scripts

To execute the Python scripts for quantizing the models, simply run the following command:

```bash
python <script_name>.py
```

### Note:
Make sure that paths to the teacher model and path to save the model are correct. The paths should point to the models located in the `/ResnetModels` folder (or any other directory where your models are stored).