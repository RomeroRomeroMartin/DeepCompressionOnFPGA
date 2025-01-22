# Pruning the Base Model
This directory contains resources and scripts related to the pruning of the original **ResNet-50** model. 

## Folder Structure
### 1. **[.vai](./.vai)**
This hidden folder contains results and information generated automatically during the pruning process. It stores details such as pruning metrics and configurations applied to the models.

### 2. **[models](./models)**
This folder contains the original **ResNet-50** (`teacher.pth`) and all the pruned models. 

### 3. **[cg_pruning.py](./cg_pruning.py)**
A Python script that executes the pruning of the base model.

### 4. **[test.py](./test.py)**
A Python script to perform an accuracy test of a model.


## How to run the scripts

To execute the Python script for pruning the models, simply run the following command:

```bash
python cg_pruning.py
```
### Note:

Make sure to update the paths to the models inside the ``cg_pruning.py`` script according to the model you want to prune. The paths should point to the model located in the ``/models`` folder (or any other directory where your models are stored). In the script you can also change the type of pruning applied (you can select `one_step` or `iterative`)
