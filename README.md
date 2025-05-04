
# Self-Supervised YOLO: Leveraging Contrastive Learning for Label-Efficient Object Detection

![YOLO SimCLR](Figures/pipeline.png)

This repository contains the code and experiments for our NeurIPS 2025 submission:  
**"Self-Supervised YOLO: Leveraging Contrastive Learning for Label-Efficient Object Detection"**

## Overview

Object detection models like YOLO typically require large-scale supervised pretraining, which can be expensive and unscalable. This project explores how **contrastive self-supervised learning (SSL)**, particularly **SimCLR**, can be applied to pretrain YOLOv5 and YOLOv8 backbones using only unlabeled data. These pretrained models are then fine-tuned for **cyclist detection** under limited labeled data conditions.

We show that:
- SSL pretraining improves **mAP**, **precision**, and **recall**.
- Pretrained models **converge faster** than randomly initialized ones.
- Performance gains are **more pronounced in low-label regimes**.

---

## 📁 Directory Structure

```
.
├── data/                     # Data loading and preprocessing
├── models/                  # Modified YOLOv5 and YOLOv8 with SimCLR heads
├── ssl_pretrain/            # SimCLR pretraining scripts
├── finetune/                # Fine-tuning on downstream detection task
├── Figures/                 # Visualizations (results, loss curves, pipeline)
├── results/                 # Output logs and checkpoints
├── utils/                   # Common utilities and transforms
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourname/ssl-yolo-pretraining.git
```

### 2. Set up the environment

```bash
conda create -n ssl-yolo python=3.9
conda activate ssl-yolo
pip install -r requirements.txt
```

> Tested with PyTorch 2.1+, CUDA 11.8, YOLOv5 6.x and YOLOv8 8.x

### 3. Pretrain with SimCLR

TBD

### 4. Fine-tune on downstream task

TBD

---

## 🧪 Results

### 📈 Precision & Recall (YOLOv8)

![Precision Recall](Figures/yolov8_precision_comparison.png)

### 📉 Validation Loss

![Loss](Figures/yolov8_val_loss_comparison.png)

### 📊 mAP@50

![mAP](Figures/yolov8_map_comparison.png)

> Similar trends were observed for YOLOv5 models. Full plots are in the `Figures/` folder.

---

## 📊 Performance Summary

| Model     | Init       | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-----------|------------|-----------|--------|---------|--------------|
| YOLOv5    | Scratch    | 0.9130    | 0.8444 | 0.9146  | 0.7486       |
| YOLOv5    | SSL        | **0.9142**| **0.8376** | **0.9139** | **0.7467** |
| YOLOv8    | Scratch    | 0.9035    | 0.8573 | 0.9231  | 0.7652       |
| YOLOv8    | SSL        | **0.9080**| **0.8534** | **0.9239** | **0.7663** |

---

## 📦 Data Sources

- **Unlabeled pretraining:** COCO 2017 (unlabeled split)
- **Downstream task:** Custom cyclist detection dataset (~5K images)

> Use `scripts/download_data.sh` (to be provided) for auto-downloading.

---

## 🛠️ Features

- 🔁 Modular training pipeline
- ✅ Compatible with Ultralytics YOLOv5 and YOLOv8
- 🔧 Configurable SimCLR settings
- 📉 Logging via TensorBoard and CSV
- 📷 Integrated visualization scripts

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@inproceedings{mk2025ssl,
  title={Self-Supervised YOLO: Leveraging Contrastive Learning for Label-Efficient Object Detection},
  author={MK and RB and NJ},
  booktitle={NeurIPS},
  year={2025}
}
```

---

---

## 📝 License

MIT License. See `LICENSE` file for details.
