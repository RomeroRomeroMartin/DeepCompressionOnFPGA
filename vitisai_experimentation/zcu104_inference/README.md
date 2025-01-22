# Code for deploy models in ZCU104

This directory contains the resources to deploy the models in the FPGA ZCU104.

## Structure Folder

## 1. **[xmodels_compiled](./xmodels_compiled)**
This folder contains all the compiled models. These compiled models can be also found in the different experimentation folders of `vitisai_experimentation`. Specifically, in each folder of quantization experiments with Vitis Ai, exits a folder named `compiled_model`, the models inside these folders (`compiled_model`) are the same as the models in this folder (`xmodels_compiled`)

## 2. **[run_inference.py](./run_inference.py)**
A ptyhon script to run the inference with the compiled models. It returns the accuracy of the model and the mean velocity of inference in FPS.


## How to run the script

To execute the Python script for the inference, simply run the following command:

```bash
python run_inference.py --image_dir <dataset_path> --model <model_path>
```

## Note:

The dataset also needs to be on the board, and it has to have the same structure as [cifar10_images](../cifar10_images/). You can add this folder on the board, or if your board has an internet connection, you can run the program [dataset.py](../dataset.py) and download it directly on the board itself.