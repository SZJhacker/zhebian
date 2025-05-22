# ZheBian（赭鞭）: a novel framework integrating sparse selection and attention for crop genome-to-phenotype prediction

## 🌾 Introduction

**Zhebian**  is a novel framework integrating sparse selection and attention for crop genome-to-phenotype prediction. The name is inspired by Shennong (神农), the mythical founder of Chinese agriculture and medicine, who used ‘zhebian’ to **identify the properties of herbs**. We aim to evoke the metaphor of Shennong’s empirical discovery process—now reimagined through modern omics and computational tools to efficiently analyze and predict complex crop traits, unlocking genetic potential and promoting precision agriculture.  It integrates sparse feature selection, sequential modeling, and attention-based fusion.

---
## 📦 1 Installation & Environment Setup

### 📁 1.1 Clone the Repository
```bash
git clone https://github.com/yourusername/zhebian.git
cd zhebian
```

You can use either a Conda environment or Docker to set up the tool.

###  1.2 Environment
### 🧪 Option 1 Conda Environment 
```bash
conda create -n zhebian python=3.9
conda activate zhebian
pip install -r requirements.txt
```

### 🐳 Option 2 Docker Build

ZheBian supports deployment via Docker, making it easy to run without manually setting up the environment. Two versions are available: one for **CPU** and one for **GPU** (requires NVIDIA GPU support).

#### Build the CPU Version
```bash
docker build -f Dockerfile.cpu -t zhebian-cpu .
```

#### Build the GPU Version (with NVIDIA GPU support)
Make sure [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed:
```bash
docker build -f Dockerfile.gpu -t zhebian-gpu .
```

## ⚙️ 2 Usage Instructions

### 2.1 Training and Prediction with conda
```bash
### Training
python zhebian.py \
  -i train.csv \
  -l Trait_Name \
  -o zhebian \


### Prediction
python predict.py \
  -l Trait_Name \
  -m zhebian \
  -i test.csv \
  -o test_out.csv
```

### 2.2 Training and Prediction with Docker

####  Train a Model (CPU)
```bash
docker run --rm -v $(pwd):/app zhebian-cpu \
  python zhebian.py --data train.csv --label "TraitName" -o zhebian
```

#### Train a Model (GPU)
```bash
docker run --rm --gpus all -v $(pwd):/app zhebian-gpu \
  python zhebian.py --data train.csv --label "TraitName" -o zhebian
```

#### Make Predictions Using a Trained Model
```bash
docker run --rm -v $(pwd):/app zhebian-cpu \
  python predict.py \
  -l Trait_Name \
  -m zhebian \
  -i test.csv \
  -o test_out.csv
```
Replace `zhebian-cpu` with `zhebian-gpu` to use GPU support.


## 📁 3 File Descriptions

| File Name              | Description |
|------------------------|-------------|
| `zhebian.py`           | Main training script combining GRU, Transformer, and LassoCV. |
| `predict.py`           | Script for making predictions using trained models. |
| `train.csv`            | Example input data for training. |
| `test.csv`             | Example input data for prediction. |
| `test_out.csv`         | Output file with prediction results. |
| `zhebian_model.h5`        | Saved trained GRU-Transformer model. |
| `zhebian_selector.joblib` | Saved trained LassoCV feature selector. |
| `zhebian_scaler.joblib` | Saved trained Scaler model. |










