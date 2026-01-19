# Plant Disease Detection

A machine learning project that uses convolutional neural networks to detect plant diseases from leaf images. The system includes both training scripts and a user-friendly GUI application.

## Features

- **CNN-based Disease Classification**: Trains a deep learning model to identify plant diseases
- **GUI Application**: Simple Tkinter interface for easy image upload and prediction
- **PlantVillage Dataset Support**: Compatible with the PlantVillage dataset structure

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Pillow (PIL)
- Tkinter (usually included with Python)

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install tensorflow numpy pillow
   ```

## Dataset Setup

1. Download the PlantVillage dataset
2. Extract it to a folder named `plantVillage` in the project root
3. Ensure the dataset structure follows:
   ```
   plantVillage/
   ├── class1/
   ├── class2/
   └── ...
   ```

## Usage

### Training the Model

Run the training script:
```bash
python train.py
```

This will:
- Load images from the `plantVillage` directory
- Train a CNN model for 5 epochs
- Save the trained model as `plant_disease_manual.h5`

### Running the GUI Application

After training, launch the GUI:
```bash
python gui.py
```

The GUI allows you to:
- Upload leaf images
- View the prediction results
- See the processed image

## Model Architecture

The CNN consists of:
- 2 Convolutional layers with ReLU activation
- Max pooling layers
- Dense layers with softmax output for classification
