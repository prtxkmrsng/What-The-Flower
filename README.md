# What The Flower 🌻

**A Biodiversity Intelligence Application**

Welcome to the **What The Flower** project, a mobile-vision repository dedicated to biodiversity intelligence and plant identification. Developed completely locally, this project leverages a lightweight, highly efficient deep learning architecture tailored for edge-AI and mobile use cases to identify various plant species.

## 🧠 Model Details

The core of this project relies on a fine-tuned vision model built using PyTorch:

* **Architecture:** MobileNetV3 Large (`models.mobilenet_v3_large`). This architecture was chosen for its optimal balance between performance and computational efficiency, making it ideal for mobile vision tasks.
* **Dataset:** iNaturalist 2021 (`2021_train_mini` for training and `2021_valid` for validation).
* **Output Classes:** Dynamically mapped based on the iNaturalist dataset categories and exported as `class_names.json` for inference.
* **Weights:** The model initializes with `MobileNet_V3_Large_Weights.DEFAULT` before the final classifier layer is replaced and fine-tuned for the specific number of plant classes.

## 📊 Training Statistics & Configuration

The training pipeline includes several advanced mechanisms to optimize performance and prevent overfitting:

* **Hyperparameters:**
* **Batch Size:** 128
* **Learning Rate:** 0.0001 (Optimized for fine-tuning)
* **Optimizer:** Adam
* **Loss Function:** CrossEntropyLoss


* **Hardware Acceleration:** Utilizes PyTorch Automatic Mixed Precision (AMP) with `torch.amp.autocast('cuda')` and `GradScaler` for faster, memory-efficient GPU training.
* **Metrics Tracked:** * Loss (Training and Validation)
* Top-1 Accuracy
* Top-5 Accuracy


* **Data Augmentation:** RandomResizedCrop(224), RandomHorizontalFlip, and ColorJitter (brightness/contrast/saturation at 0.2).
* **Fault Tolerance & "Second Wind" Logic:** * Custom `EarlyStopper` with a base patience of 10 epochs.
* **Second Wind:** If the validation loss plateaus (>= 1.0) before epoch 95, the script automatically drops the learning rate by a factor of 10, increases patience by 5, and reloads the best saved weights to escape the local minimum.
* **Resumption:** Native support for resuming training from a specific epoch using the saved `best_plant_model.pth` state dictionary.



## 🚀 Step-by-Step Guide: How to Run and Test

Follow these steps to replicate the training environment, run the model, and monitor its progress.

### 1. Prerequisites

Ensure you have Python installed along with the required libraries.

```bash
pip install torch torchvision tensorboard

```

### 2. Prepare the Environment

Clone the repository and navigate to the `mobile_vision` directory. The training scripts automatically handle the downloading and extraction of the iNaturalist dataset if it is not present in the `./data` directory.

### 3. Start Training

You can initiate the training process using the standard script. If you need to stop and resume later, the script is configured to pick up from the last saved state (`best_plant_model.pth`).

```bash
python train_mobilenet.py

```

*Note: The script is configured to use CUDA by default if available. Ensure your NVIDIA drivers are up to date to leverage GPU acceleration.*

### 4. Monitor with TensorBoard

The training script logs detailed metrics (Loss, Top-1 Accuracy, and Top-5 Accuracy) to TensorBoard. To visualize the training process in real-time, open a new terminal window and run:

```bash
tensorboard --logdir=./runs/inaturalist_mobilenet_v3

```

Then, open your web browser and navigate to `http://localhost:6006`.

### 5. Testing and Inference

Once training is complete (or stopped via early stopping), the best weights are saved to `best_plant_model.pth`. A `class_names.json` file is also generated, mapping the model's output indices to human-readable plant categories.

To use the model for inference in your own mobile application or testing scripts:

1. Load a `mobilenet_v3_large` architecture.
2. Modify the final classifier layer to match the length of `class_names.json`.
3. Load the state dictionary from `best_plant_model.pth`.
4. Pass your images through the same validation transforms (Resize 256, CenterCrop 224, Normalize).
