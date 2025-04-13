# 🔍 SimCLR-YOLO: Self-Supervised Pretraining for YOLO Backbones

> Self-supervised contrastive learning using SimCLR on a YOLO-style CSPDarknet backbone with COCO unlabeled images.

This repository provides a full pipeline for pretraining a YOLO-compatible backbone (CSPDarknet) using SimCLR on unlabeled image datasets like COCO. It is designed to improve YOLO performance with minimal labels, and to serve as a research-ready template for self-supervised object detection.

---

## ✅ ML Code Completeness Checklist

- [x] Specification of dependencies (`requirements.txt` / `environment.yml`)
- [x] Training code (`simclr_training.py`)
- [ ] Evaluation code (to be added for feature quality evaluation)
- [x] Pre-trained models (link below)
- [x] README with results + instructions

---

## 🧪 Results

| Epoch | Loss    | Pretrained Weights                                                   |
|-------|---------|------------------------------------------------------------------------|
| 50    | ~3.94   | [Download](https://huggingface.co/your-model-path-or-gdrive-link)     |
| 75    | ~3.92   | [Download](https://huggingface.co/your-model-path-or-gdrive-link)     |
| 100   | ~3.91   | [Download](https://huggingface.co/your-model-path-or-gdrive-link)     |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/manikantakotthapalli/SSL-YOLO-Pretraining.git
cd SSL-YOLO-Pretraining
```

### 2. Setup environment
Create a conda environment or use `requirements.txt`:
```bash
conda create -n ssl-yolo python=3.10 -y
conda activate ssl-yolo
pip install -r requirements.txt
```

### 3. Prepare data
Download COCO unlabeled dataset and place images under:
```
data/coco_unlabeled/dummy_class/*.jpg
```

### 4. Run SimCLR pretraining
```bash
python simclr_training.py
```
This will train CSPDarknet using SimCLR and save weights at `ssl_cspdarknet_epoch75.pth`.

---

## 🧠 Pretrained Weights

| Model                   | Epoch | Link                                                                     |
|------------------------|-------|--------------------------------------------------------------------------|
| CSPDarknet (SimCLR)    | 50    | [ssl_cspdarknet_epoch50.pth](https://huggingface.co/your-model-link)     |
| CSPDarknet (SimCLR)    | 75    | [ssl_cspdarknet_epoch75.pth](https://huggingface.co/your-model-link)     |
| CSPDarknet (SimCLR)    | 100   | [ssl_cspdarknet_epoch100.pth](https://huggingface.co/your-model-link)    |

---

## 📦 Repo Structure
```
.
├── models/
│   └── cspdarknet.py         # YOLO-style backbone
├── simclr_training.py        # Main training script
├── augmentations.py          # SimCLR-style data aug (TBD)
├── requirements.txt
├── data/
│   └── coco_unlabeled/       # Unlabeled image dataset
└── README.md
```

---

## 📝 Citation
If you use this code in your research, please cite:
```
@misc{simclr_yolo2025,
  title={SimCLR-YOLO: Self-Supervised Pretraining for YOLO Backbones},
  author={Manikanta Kotthapalli},
  year={2025},
  howpublished={\url{https://github.com/manikantakotthapalli/SSL-YOLO-Pretraining}}
}
```

---

## 📬 Contact
For questions or suggestions, open an issue or contact [@manikantakotthapalli](https://github.com/manikantakotthapalli)

---

## 📘 License
This project is licensed under the MIT License.
