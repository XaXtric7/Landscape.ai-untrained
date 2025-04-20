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

## 📷 Screenshots

![Image](https://github.com/user-attachments/assets/6c55636b-ea50-4f82-8ac6-8412e1a269f4)

![Image](https://github.com/user-attachments/assets/65363ada-a8be-49f1-be0f-88b56ebd2e73)

![Image](https://github.com/user-attachments/assets/ade934ed-4807-40ea-b656-646e3cea51b2)

![Image](https://github.com/user-attachments/assets/1222ebe3-2827-4518-b31a-9fb5ce22d4cd)

![Image](https://github.com/user-attachments/assets/8a7e6c02-8bfa-47ef-9ec5-59257ecf2c3e)

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

## 🐳 Docker Support

### Build and Run in Docker

```bash
docker build -t segment_project_image .
docker run --name segment_container -d segment_project_image
```

### Copy Outputs

```bash
docker cp segment_container:/segment_project/models ./models
docker cp segment_container:/segment_project/logs ./logs
docker cp segment_container:/segment_project/output ./output
```

### Cleanup

```bash
docker stop segment_container
docker rm segment_container
docker rmi segment_project_image

```

## 📁 Project Structure

```bash
├── config/
│   └── config.yaml        # Configuration file (classes, model params)
├── data/
│   ├── train/
│   ├── test/
├── models/                # Pretrained weights
├── output/
│   ├── predicted_masks/   # .tif mask outputs
│   ├── prediction_plots/  # .png visualizations
├── logs/                  # Training and testing logs
├── src/
│   ├── train.py
│   ├── test.py
│   └── inference.py
├── requirements.txt
└── README.md

```

### 🔧 Technologies Used

## 🧠 Deep Learning & Vision

- PyTorch 2.0.1
- torchvision 0.15.2
- segmentation_models_pytorch 0.3.3
- EfficientNet (efficientnet-pytorch)
- timm (PyTorch Image Models)

## 🧰 Image & Data Handling

- OpenCV (opencv-python-headless)
- patchify
- scikit-image, scikit-learn
- albumentations
- tifffile, Pillow, NumPy

## 📦 Utilities

- PyYAML
- tqdm
- split-folders
- filelock, joblib
- pretrainedmodels
- huggingface-hub

## 📈 Loss and Optimization

- Dice Loss
- Adam Optimizer
- LR Scheduler (PyTorch)

## 🐳 Deployment

- Docker

## 📜 License

This project is released under the MIT License.
