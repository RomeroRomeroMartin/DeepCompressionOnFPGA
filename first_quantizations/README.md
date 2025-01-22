# Training and Quantization of ResNet-50 with Pytorch

This directory contains resources and scripts related to the training and quantization of Simple Model.

## Folder Structure
### 1. **[models](./models)**
This folder contains the original model and all the quantized models. 

### 2. **[first_quantizations_pytorch.ipynb](./first_quantizations_pytorch.ipynb)**
A Jupyter Notebook that performs the training and quantization of the Simple Model. It also generates the figures with information of the models.

### 3. **[test.py](./test.py)**
A Python script to perform an accuracy test of a model.


## How to run the script

To execute the Python script for testing the models, simply run the following command:

```bash
python test.py
```
### Note:

Make sure to update the paths to the models inside the ``test.py`` script according to the model you want to test. The paths should point to the model located in the ``/models`` folder (or any other directory where your models are stored). 