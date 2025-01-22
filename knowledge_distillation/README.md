# Distillation of ResNet-50
This directory contains resources and scripts related to the distillation of the original **ResNet-50** model.

## Folder Structure
### 1. **[models](./models)**
This folder contains the original **ResNet-50** (`teacher.pth`) and all the distilled models. 

### 2. **[common.py](./common.py)**
A Python script with auxiliar code.

### 3. **[get_models_info.ipynb](./get_models_info.ipynb)**
A Jupyter Notebook with all the figures related to the distillation of the ResNet-50.

### 4. **[kd_resnet.py](./kd_resnet.py)**
A Python script to distillate a ResNet-50 to a models from scratch.

### 5. **[resnet2mobilenet.py](./resnet2mobilenet.py)**
A Python script to distillate a ResNet-50 to a MobileNet.


### 6. **[resnet502resnet18.py](./resnet502resnet18.py)**
A Python script to distillate a ResNet-50 to a ResNet-18.


### 7. **[resnet_inter_distill.py](./resnet_inter_distill.py)**
A Python script to distillate a ResNet-50 to a ResNet-18 with distillation inter layers.


### 8. **[squeezenet.py](./squeezenet.py)**
A Python script to distillate a ResNet-50 to a SqueezeNet.


### 9. **[test.py](./test.py)**
A Python script to perform an accuracy test of a model.



## How to run the scripts

To execute the Python scripts for distill the models, simply run the following command:

```bash
python <script_name>.py
```

### Note:
Make sure that paths to the teacher model and path to save the model are correct. The paths should point to the models located in the ``/models`` folder (or any other directory where your models are stored).