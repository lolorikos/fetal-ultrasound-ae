# Fetal Ultrasound Auto-Encoder

A deep learning project for learning compressed representations of 
fetal ultrasound images using convolutional auto-encoders, with a 
downstream classifier for standard scan plane detection.

## Dataset

[FETAL_PLANES_DB](https://zenodo.org/records/3904280) — 12,400 labeled 
maternal-fetal ultrasound images across 6 classes:
- Fetal abdomen
- Fetal brain  
- Fetal femur
- Fetal thorax
- Maternal cervix
- Other

## Project Structure

    fetal-ultrasound-ae/
    ├── src/
    │   ├── dataset.py    # custom Dataset class
    │   ├── model.py      # Encoder, Decoder, AutoEncoder, Classifier
    │   ├── train.py      # training loops, save/load, visualisation
    ├── notebooks/
    │   └── exploration.ipynb  # EDA and training experiments
    ├── results/           # saved model weights
    └── requirements.txt

## Results

### Stage 1 — Auto-Encoder Reconstruction
Trained for 20 epochs on 7,129 images. 16x compression ratio.

| Input | Compressed | Reconstructed |
|-------|-----------|---------------|
| 1×128×128 = 16,384 numbers | 16×8×8 = 1,024 numbers | 1×128×128 |

### Stage 2 — Scan Plane Classification
Fine-tuned encoder + classification head. Trained with weighted loss 
to handle class imbalance.

| Class | F1 Score |
|-------|----------|
| Fetal abdomen | 0.45 |
| Fetal brain | 0.91 |
| Fetal femur | 0.75 |
| Fetal thorax | 0.80 |
| Maternal cervix | 0.96 |
| Other | 0.79 |
| **Overall accuracy** | **81%** |

## Setup

```bash
git clone https://github.com/lolorikos/fetal-ultrasound-ae
cd fetal-ultrasound-ae
pip install -r requirements.txt
```

Download the dataset from [Zenodo](https://zenodo.org/records/3904280) 
and place it in the `data/` folder.

## Approach

1. **Unsupervised pre-training** — train a convolutional auto-encoder 
   to learn compressed representations of ultrasound images without labels
2. **Supervised fine-tuning** — add a classification head and fine-tune 
   the entire network using labelled scan plane data
3. **Class imbalance handling** — weighted CrossEntropyLoss to handle 
   the heavily imbalanced class distribution

## Requirements
- Python 3.11+
- PyTorch
- torchvision
- pandas
- scikit-learn
- matplotlib
- tqdm
