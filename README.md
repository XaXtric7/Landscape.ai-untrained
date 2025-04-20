# Terra_Mask 🛰️

**Land Cover Semantic Segmentation using PyTorch**

---

## 📌 What Is This Project?

**Terra_Mask** is an end-to-end computer vision project for **semantic segmentation** in land cover classification. It uses deep learning with **PyTorch** to classify features in high-resolution satellite imagery — such as **buildings**, **woodland**, **water**, and **roads**.

Built primarily for the [LandCover.ai dataset](https://www.kaggle.com/datasets/adrianboguszewski/landcoverai), this project is modular and can be adapted to any semantic segmentation dataset.

---

## 🌍 What Does It Do?

This project enables you to:

- ✅ **Train** segmentation models on land cover images
- ✅ **Test** and **infer** with trained models
- ✅ **Prompt** specific classes for prediction without retraining

**Promptable Prediction:** After training the model on all classes, you can selectively visualize only specific classes (e.g., just "buildings" and "water") by modifying the `test_classes` in the config.

---

## 🧠 How It Works

### 1. **Data Preparation**

- Patches are created from large aerial images.
- Background-heavy patches are discarded.
- Data is split into train/val/test sets.

### 2. **Model Architecture**

- Uses **UNet** from `segmentation_models_pytorch`
- **EfficientNet** as the encoder backbone
- Decoder upsamples features to produce segmentation masks

### 3. **Training**

- Uses **Dice Loss** for segmentation accuracy
- Learning rate scheduling for better optimization

### 4. **Inference**

- Patches are passed through the model
- Reconstructed masks are filtered by class
- Visualizations are generated and saved

---

## 🚀 Quick Start

### 📁 Dataset Setup

Copy the dataset from [LandCover.ai on Kaggle](https://www.kaggle.com/datasets/adrianboguszewski/landcoverai)

Organize it as:

```bash
data/
├── train/
│ ├── images/ # Training satellite images
│ └── masks/ # Corresponding segmentation masks
└── test/
├── images/ # Testing satellite images
└── masks/ # Corresponding segmentation masks
```

### 🔧 Environment Setup

```bash
# Clone this repository
git clone https://github.com/souvikmajumder26/Land-Cover-Semantic-Segmentation-PyTorch.git
cd Land-Cover-Semantic-Segmentation-PyTorch

# Install required packages
pip install -r requirements.txt
```

## 🏃‍♂️ How to Run

### 🔬 Testing with Pre-trained Model

```bash
cd src
python test.py
```

### 🎯 Inference

```bash
cd src
python inference.py
```

### 🏋️‍♂️ Training the Model(⚠️ Requires significant GPU resources.)

```bash
cd src
python train.py
```
