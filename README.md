# Model Compression and Deployment for Edge Devices

This repository contains the implementation of techniques for compressing deep learning models and preparing them for deployment on edge devices, specifically using the Xilinx ZCU104 FPGA. The goal of this project is to explore various methods for optimizing model performance while maintaining acceptable accuracy. 

## Overview of the Project

The project focuses on the following key objectives:

1. **Model Quantization**: Reducing the precision of model weights and activations to lower bit-widths (e.g., 8-bit, 5-bit) to reduce memory usage and improve inference speed.
2. **Pruning**: Removing less significant weights or neurons in the network to reduce model size and computational complexity.
3. **Knowledge Distillation**: Transferring knowledge from a larger, more accurate teacher model to a smaller student model, which is easier to deploy on resource-constrained devices.
4. **Deployment Using Vitis-AI**: Preparing compressed models for deployment on the Xilinx ZCU104 FPGA, which includes quantization, pruning, and compilation using the Vitis-AI framework.

## Repository Structure

The repository is organized into four main directories:

### 1. **[Simple Quantization](./first_quantizations)**
   - **Purpose**: Initial experiments with model quantization on simpler architectures to validate the process.
   - **Contents**: Includes scripts for quantizing basic models and analyzing results.
   - **Readme**: Provides detailed instructions on folder structure and usage.

### 2. **[ResNet-50 Quantization](./resnet50_quantization)**
   - **Purpose**: Quantization experiments specifically on the ResNet-50 model, including post-training quantization (PTQ) and Quantization Aware Training (QAT).
   - **Contents**: Contains the quantized models, evaluation scripts, and results.
   - **Readme**: Explains how to execute the quantization process and interpret the results.

### 3. **[ResNet-50 Distillation](./knowledge_distillation)**
   - **Purpose**: Application of knowledge distillation techniques to ResNet-50.
   - **Contents**: Includes distilled models, scripts, and results for MobileNet, ResNet-18, and SqueezeNet students.
   - **Readme**: Guides the user through the distillation process.

### 4. **[Vitis-AI Workflow](./vitisai_experimentation)**
   - **Purpose**: Integrates pruning, quantization, and compilation processes using the Vitis-AI framework to prepare models for deployment on the ZCU104 FPGA.
   - **Contents**: Contains scripts for pruning, quantization, compilation, and deployment, along with the resulting models and logs.
   - **Readme**: Details the structure and provides step-by-step instructions for using the Vitis-AI workflow.

## How to Get Started

1. **Install Dependencies**: Follow the individual Readmes in each directory to set up the required dependencies for that workflow.
2. **Run Experiments**: Use the provided scripts to replicate the quantization, pruning, or distillation processes.
3. **Deploy Models**: Follow the instructions in the Vitis-AI workflow directory to deploy models on the ZCU104 FPGA.

## Key Contributions

- **Compression Techniques**: Explored multiple techniques (quantization, pruning, and distillation) to compress deep learning models.
- **Adaptation to Edge Devices**: Focused on preparing models for deployment on resource-constrained devices, with special emphasis on FPGA deployment.
- **Comprehensive Documentation**: Provided detailed Readmes for each workflow to ensure reproducibility and clarity.

## Future Work

- Experimenting with additional compression techniques, such as structured pruning and mixed-precision quantization.
- Extending deployment to other edge devices, including GPUs and embedded CPUs.
- Comparing the performance of the compressed models across different platforms.

## Acknowledgments

This project has been developed as part of a Master's Thesis, exploring cutting-edge techniques for deep learning model compression and deployment on edge devices. Special thanks to the tools and frameworks that made this work possible, including PyTorch, Vitis-AI, and Docker.

## Author

Martín Romero Romero

---

For specific details, refer to the Readme files within each directory.
