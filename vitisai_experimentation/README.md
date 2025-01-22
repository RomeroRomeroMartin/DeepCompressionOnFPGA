# Code for Using Vitis-AI

This directory contains the necessary code to run a model using the **Vitis-AI** platform. It includes scripts and resources to prune, quantize and compile  Deep Learning models to be deployed on an FPGA, such as the **ZCU104**.

## Installation

To use this code, make sure the **Vitis-AI** environment is pre-configured and that you are working within a compatible container. Use the following commands to download and activate the environment:
All the steps can be found in: https://xilinx.github.io/Vitis-AI/3.5/html/docs/install/install.html

Previous requirements:

    - CUDA (https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
    
    - Docker (https://docs.docker.com/engine/install/)

Once CUDA and Docker are installed:
```bash
sudo apt-get update

```
Clone the repository:
```bash
git clone https://github.com/Xilinx/Vitis-AI

```
Move to the path to install the docker
```bash
cd <Vitis-AI install path>/Vitis-AI/docker
```
Install the image according your preferences
```bash
./docker_build.sh -t <DOCKER_TYPE> -f <FRAMEWORK>

```
In our case:
```bash
./docker_build.sh -t gpu -f pytorch

```
You have all this options to install the images:

| **DOCKER_TYPE (-t)** | **TARGET_FRAMEWORK (-f)** | **Desired Environment**                      |
|-----------------------|---------------------------|----------------------------------------------|
| `cpu`                | `pytorch`                | PyTorch cpu-only                             |
|                      | `tf2`                    | TensorFlow 2 cpu-only                        |
|                      | `tf1`                    | TensorFlow 1.15 cpu-only                     |
| `gpu`                | `pytorch`                | PyTorch with AI Optimizer CUDA-gpu           |
|                      | `tf2`                    | TensorFlow 2 with AI Optimizer CUDA-gpu      |
|                      | `tf1`                    | TensorFlow 1.15 with AI Optimizer CUDA-gpu   |
| `rocm`               | `pytorch`                | PyTorch with AI Optimizer ROCm-gpu           |
|                      | `tf2`                    | TensorFlow 2 with AI Optimizer ROCm-gpu      |


# Move Project Files to Vitis-AI Directory
Once the docker image is installed, we can move all the contents of this folder inside de Vitis-AI folder 

```bash
mv <path to this folder>/vitis_repo <Vitis-AI path>/Vitis-AI

```
# Running the Docker Container

To launch the Docker container with the appropriate environment and access to your project files, use the following command:

```bash
cd <Vitis-AI install path>/Vitis-AI
./docker_run.sh xilinx/vitis-ai-<pytorch|tensorflow2|tensorflow>-<cpu|gpu|rocm>:latest
```
If the name or tag of the image is different, you can execute this command to see the list of docker images:
```bash
docker images
```

**Once you can run the docker container, you can start to execute the programs to quantize, prune and compile models using Vitis AI**

# Contents of this folder
This project is organized into several directories, each containing the relevant files for different stages of model quantization, pruning, and compilation for deployment on the FPGA. Below is an overview of each folder's content:

## 1. **[cifar10_images](./cifar10_images)**
This folder contains the CIFAR-10 dataset, which is commonly used for training and evaluating machine learning models. The dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. These images are used for tasks such as classification and model evaluation. 
### Folder structure:
- **Each subfolder** corresponds to one of the 10 classes in the CIFAR-10 dataset. The subfolders contain the images belonging to that particular class, making it easy to organize and access the data for training and evaluation.

The classes are as follows:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## 2. **[distill_pruning_quantization](./distill_pruning_quantization)**

This folder contains scripts and resources for applying model pruning, and quantization processes to previous distilled models (MobileNet, ResNet-18 and SqueezeNet). 


## 3. **[pruning](./pruning)**

This folder contains scripts to apply pruning methods to the base model (ResNet-50). 

## 4. **[quantization](./quantization)**
This folder contains scripts and resources to apply quantization to the base model (ResNet-50). Both Pos-Training Quantization (PTQ) and Quantization Aware Training (QAT) were applied.

## 5. **[quantization_distillation](./quantization_distillation)**
This folder contains scripts and resources to apply quantization to previous distilled models (MobileNet, ResNet-18 and SqueezeNet). Only PTQ was applied in this folder.

## 6. **[quantization_pruning](./quantization_pruning)**
This folder contains scripts and resources to apply quantization to the previous pruned base model. (The best models of "pruning" folder are quantized here). Only PTQ was applied in this folder

## 7. **[zcu104_inference](./zcu104_inference)**
This folder contains the models and the python script to be executed in the FPGA ZCU104. 

## 8. **[get_models_info.ipynb](./get_models_info.ipynb)**
This folder contains a Jupyter Notebook that generates visualizations and graphs to support the analysis of the models in the project. The notebook is designed to work with the models present in the various folders, including those that have undergone distillation, pruning, and quantization.

### Purpose:
- The notebook generates key metrics and graphs, such as accuracy, model size, inference time, and other performance indicators, to evaluate the effectiveness of different optimization techniques (distillation, pruning, quantization) applied to the models.
- The generated graphs and results are intended to be included in the final report (thesis or documentation), offering a clear comparison between the models before and after optimization.

### Note:
- Please be aware that the paths to the models may have changed. If you wish to execute the notebook, make sure to review and update the paths to the models accordingly.

## 9. **[dataset.py](./dataset.py)**
This python program download de CIFAR10 dataset ("cifar10_images" folder)

