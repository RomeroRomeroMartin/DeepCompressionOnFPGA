import os
import threading
import time
import numpy as np
import xir
import vart
from ctypes import *
from typing import List
import sys
import argparse
import pathlib
import cv2
from PIL import Image

# Definir la función de preprocesamiento
def preprocess_fn(image_path, fix_scale):
    # Leer la imagen
    image = cv2.imread(image_path)
    # Redimensionar la imagen a 224x224
    image = image[:, :, ::-1]
    image = cv2.resize(image, (224, 224))

    # Normalizar la imagen a rango [0, 1]
    image = image / 255.0

    # Aplicar la normalización similar a PyTorch
    image = (image - np.array([0.5, 0.5, 0.5])) / np.array([0.5, 0.5, 0.5])

    # Escalar según el factor de cuantización y convertir a int8
    image = (image * fix_scale).astype(np.int8)
    return image

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

# Función para obtner la clase predicha
def runDPU(dpu, img):
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)

    # Obtener el fix_point de la salida y calcular el factor de escalado
    output_fixpos = outputTensors[0].get_attr("fix_point")
    output_scale = 1 / (2 ** output_fixpos)

    batchSize = input_ndim[0]
    n_of_images = len(img)
    count = 0
    ids = []
    ids_max = 10
    outputData = [np.empty(output_ndim, dtype=np.int8, order="C") for _ in range(ids_max)]
    global out_q
    out_q = [None] * n_of_images  # Prealoca el vector de salida

    runSize = min(batchSize, n_of_images - count)
    # Preparar los datos de entrada
    inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]
    inputData[0][0, ...] = img[0].reshape(input_ndim[1:])
    # Ejecutar la DPU
    job_id = dpu.execute_async(inputData[0], outputData[0])
    ids.append((job_id, runSize, count))

    # Procesar los resultados si alcanzamos el límite de trabajos
    # Esperar los trabajos en cola y procesar las salidas
    dpu.wait(job_id)
    scaled_output = outputData[0][0] * output_scale
    out_q = np.argmax(scaled_output)  # Obtener la clase predicha
    return [out_q]

# Función principal
def app(image_dir, model):
    classes = os.listdir(image_dir)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print(classes)

    # Cargar el modelo
    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)

    runner = vart.Runner.create_runner(subgraphs[0], "run")
    input_fixpos = runner.get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2 ** input_fixpos
    global out_q
    out_q = [None]

    total_images = 0
    correct_predictions = 0
    total_times = []
    for cls in classes:
        cls_path = os.path.join(image_dir, cls)
        if os.path.isdir(cls_path):
            for img_file in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_file)
                img = preprocess_fn(img_path, input_scale)
                inicio = time.time()                
                prediction_index = runDPU(runner, [img])
                fin = time.time()
                total_times.append(fin - inicio)
                prediction_class = classes[prediction_index[0]]
                print(img_path)
                print('Prediction: ', prediction_class, 'Real: ', cls)
                print('\n')
                if prediction_class == cls:
                    correct_predictions += 1

                total_images += 1

    accuracy = correct_predictions / total_images if total_images > 0 else 0
    print(f'Total Images: {total_images}, Correct Predictions: {correct_predictions}, Accuracy: {accuracy:.4f}, \
    Avg Time per frame: {sum(total_times) / len(total_times)}, FPS: {len(total_times) / sum(total_times)}')

    return

# Función principal
def main():
    parser = argparse.ArgumentParser(description="DPU Inference Script")
    parser.add_argument('--image_dir', type=str, required=True, help="Path to the directory containing test images.")
    parser.add_argument('--model', type=str, required=True, help="Path to the .xmodel file.")
    args = parser.parse_args()

    image_dir = args.image_dir
    model = args.model

    app(image_dir, model)

if __name__ == "__main__":
    main()