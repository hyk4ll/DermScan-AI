# Skin Disease Detection System Using Deep Learning

**Final Year Project**  
**Author:** Muhammad Haikhal Bin Omanudin Baki

## Overview

This project implements a deep learning-based skin disease detection system capable of classifying 7 types of skin lesions using the HAM10000 dataset. The system features a custom CNN architecture with hair removal preprocessing and a user-friendly web interface for real-time predictions.

## Objectives

1. **Design and develop** a deep learning-based model capable of detecting and classifying common skin diseases from image data.
2. **Evaluate** the performance and accuracy of deep learning-based models for skin disease detection.
3. **Implement** a web-based skin disease detection interface that enables users to upload skin images and view real-time prediction results.

## Detectable Skin Conditions

| Condition | Code | Description |
|-----------|------|-------------|
| Melanocytic nevi | nv | Common moles (benign) |
| Melanoma | mel | Serious skin cancer |
| Benign keratosis-like lesions | bkl | Seborrheic keratoses, solar lentigo |
| Basal cell carcinoma | bcc | Common skin cancer |
| Actinic keratoses | akiec | Pre-cancerous lesions |
| Vascular lesions | vasc | Angiomas, hemorrhage |
| Dermatofibroma | df | Benign skin growths |

## Project Structure

```
Project/
├── HAM10000_images_part_1/    # Dataset images (part 1)
├── HAM10000_images_part_2/    # Dataset images (part 2)
├── HAM10000_metadata.csv      # Dataset metadata
├── models/                    # Trained model files
│   ├── skin_disease_model.h5
│   ├── best_model.h5
│   └── class_indices.json
├── results/                   # Training results and visualizations
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── dataset_distribution.png
│   └── classification_report.txt
├── train.py                   # Training script
├── preprocessing.py           # Image preprocessing module
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd Project
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training the Model

To train the CNN model on the HAM10000 dataset:

```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Apply data augmentation and oversampling
- Train the CNN model with early stopping
- Generate evaluation metrics and visualizations
- Save the trained model to `models/`

**Training Parameters (configurable in `train.py`):**
- `IMG_SIZE`: 128x128 pixels
- `BATCH_SIZE`: 32
- `EPOCHS`: 25 (with early stopping)
- `LEARNING_RATE`: 0.0001

### 2. Running the Web Interface

After training, launch the Streamlit web application:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

**Features:**
- Upload skin lesion images (JPG, PNG)
- Real-time prediction with confidence scores
- Hair removal preprocessing visualization
- Disease information and recommendations
- All class probabilities display

## Methodology

### Data Preprocessing

1. **Hair Removal (Inpainting):**
   - Convert to grayscale
   - Apply morphological black-hat transform (17x17 kernel)
   - Create binary mask via thresholding
   - Inpaint using Telea algorithm

2. **Data Augmentation:**
   - Rotation (±20°)
   - Width/height shift (20%)
   - Shear transformation (20%)
   - Zoom (20%)
   - Horizontal flip

3. **Class Balancing:**
   - Oversampling minority classes to match majority class count

### CNN Architecture

```
Input (128x128x3)
    ↓
Block 1: Conv2D(32) → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Block 2: Conv2D(64) → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Block 3: Conv2D(128) → Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Block 4: Conv2D(256) → Conv2D(256) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Flatten → Dense(512) → BatchNorm → Dropout(0.5) → Dense(256) → Dropout(0.5)
    ↓
Output: Dense(7, softmax)
```

### Training Callbacks
- **Early Stopping:** Monitor validation loss, patience=5
- **ReduceLROnPlateau:** Reduce learning rate by 0.5 after 3 epochs of no improvement
- **ModelCheckpoint:** Save best model based on validation accuracy

## Evaluation Metrics

The model is evaluated using:
- **Accuracy:** Overall classification accuracy
- **Confusion Matrix:** Visualize prediction patterns
- **Classification Report:** Per-class precision, recall, F1-score

Results are saved in the `results/` directory.

## API Reference

### preprocessing.py

```python
# Apply hair removal to an image
from preprocessing import hair_removal
cleaned_image = hair_removal(image_path_or_array)

# Preprocess for model prediction
from preprocessing import preprocess_for_prediction
processed = preprocess_for_prediction(image, target_size=(128, 128))
```

### Disease Information

```python
from preprocessing import DISEASE_INFO
info = DISEASE_INFO['Melanoma']
print(info['description'])
print(info['severity'])
print(info['recommendation'])
```

## Troubleshooting

### Model not found error
Ensure you have trained the model first:
```bash
python train.py
```

### Out of memory during training
Reduce batch size in `train.py`:
```python
BATCH_SIZE = 16  # or 8
```

### Slow training on CPU
Consider using Google Colab with GPU or reduce image size:
```python
IMG_SIZE = 64
```

## Disclaimer

⚠️ **Important:** This system is designed for educational and screening purposes only. It should NOT be used as a substitute for professional medical diagnosis. Please consult a qualified dermatologist for accurate diagnosis and treatment.

## Dataset

This project uses the **HAM10000 dataset** (Human Against Machine with 10000 training images):
- 10,015 dermatoscopic images
- 7 diagnostic categories
- Collected over 20 years at various clinics

**Citation:**
```
Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of 
multi-source dermatoscopic images of common pigmented skin lesions. Sci Data 5, 180161 (2018).
```

## License

This project is developed as part of a Final Year Project for educational purposes.

## Contact

For questions or feedback, please contact:
- **Name:** Muhammad Haikhal Bin Omanudin Baki
- **Project:** Final Year Project (FYP)
