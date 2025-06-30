# How Many Tokens Do 3D Point Cloud Transformer Architectures Really Need?

This repository contains the official implementation, pretrained models, and evaluation code for our paper:

> **How Many Tokens Do 3D Point Cloud Transformer Architectures Really Need?**  
[![ArXiv](https://img.shields.io/badge/Paper-ArXiv-b31b1b.svg)](https://arxiv.org/pdf/2410.02615v3)
[![Hugging Face](https://img.shields.io/badge/🤗%20Model-HuggingFace-blue)](https://huggingface.co/MERGE-Group)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC--BY--NC%204.0-lightgrey.svg)](https://github.com/duyhominhnguyen/Exgra-Med/blob/main/LICENSE)


---

## 📌 Abstract

Recent advances in 3D point cloud transformers have led to state-of-the-art results in tasks such as semantic segmentation and reconstruction. However, these models typically rely on dense token representations, incurring high computational and memory costs during training and inference.

In this work, we introduce a simple and effective **token merging strategy** that reduces token counts by up to **90–95%**, with **minimal accuracy loss**. Our method leverages **spatial structural priors** in point clouds to estimate token importance, enabling aggressive reduction without sacrificing performance.

Our findings reveal that many existing 3D transformer models are **over-tokenized** and **under-optimized** for scalability. We validate our approach across multiple 3D vision tasks and demonstrate consistent **computational efficiency gains** while maintaining competitive accuracy.

---

## 🚀 Highlights

- 🔧 **Up to 95% token reduction** with negligible performance drop.
- 🧩 **Structure-aware token importance estimation**.
- 📊 Validated across multiple 3D vision tasks (semantic segmentation, reconstruction, etc.).
- 💡 Challenges assumptions about the necessity of dense tokenization in 3D transformers.

---

## 🛠️ Features

- Modular and extensible PyTorch implementation
- Support for multiple datasets (e.g., S3DIS, ScanNet, ShapeNet)
- Benchmark scripts for:
  - Semantic segmentation  
  - 3D object reconstruction  
- Token importance visualization tools
- Pretrained models and reproducibility support

---

## 📁 Repository Structure

```bash
├── models/             # Transformer and token merging models
├── datasets/           # Dataset loaders and preprocessors
├── tools/              # Training, evaluation, and token analysis scripts
├── configs/            # YAML configs for experiments
├── pretrained/         # Links or scripts for downloading pretrained weights
└── README.md
```

---
## Setup
```
git clone https://github.com/yourusername/point-transformer-token-efficiency.git
cd point-transformer-token-efficiency
pip install -r requirements.txt
```

## Training & Evaluation

Example command for training on S3DIS:

```
python tools/train.py --config configs/s3dis/token_merge.yaml
```

Evaluate pretrained model:
```
python tools/eval.py --config configs/s3dis/token_merge.yaml --checkpoint pretrained/s3dis_model.pth

```

## Results

| Task                   | Dataset  | Token Reduction | Accuracy Drop | Speedup |
|------------------------|----------|------------------|---------------|---------|
| Semantic Segmentation  | S3DIS    | 90%              | < 1%          | 2.5×    |
| Object Reconstruction  | ShapeNet | 95%              | < 2%          | 3.2×    |

## Citation
If you find this work useful, please cite:
```
@article{tran2025tokens3d,
  title={How Many Tokens Do 3D Point Cloud Transformer Architectures Really Need?},
  author={Tuan Anh Tran, Duy Minh Ho Nguyen, Hoai-Chau Tran, Michael Barz, Khoa D Doan, Roger Wattenhofer, Vien Anh Ngo, Mathias Niepert, Daniel Sonntag, Paul Swoboda},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```


